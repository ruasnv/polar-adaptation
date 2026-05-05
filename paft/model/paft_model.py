"""
PAFTAttention and PAFTModel.

Architecture overview
─────────────────────
PAFTModel wraps a GPT2LMHeadModel and replaces every GPT2Attention layer with a
PAFTAttention layer.  PAFTAttention stores the polar decomposition of each head's
W_V and W_O and reconstructs the full weight matrix on every forward pass.

Tensor shapes (GPT-2 small: n_embd=768, n_heads=12, d_head=64)
───────────────────────────────────────────────────────────────
Weight extraction (from extractor.py):
    W_V_h  [n_embd, d_head]   per-head V projection — TALL  (n_embd > d_head)
    W_O_h  [d_head, n_embd]   per-head O projection — WIDE  (d_head < n_embd)

Polar decomposition:
    W_V_h = Q_V_h @ S_V_h        right polar  (Q^T Q = I_{d_head})
    W_O_h = S_O_h @ Q_O_h        left  polar  (Q Q^T = I_{d_head})

Stacked across heads (stored as buffers/parameters):
    Q_V    [H, n_embd, d_head]   H = n_heads
    Q_O    [H, d_head, n_embd]
    S_V    [H, d_head, d_head]   symmetric PSD at init
    S_O    [H, d_head, d_head]   symmetric PSD at init
    EV_V   [H, d_head, d_head]   eigenvectors of initial S_V (columns)
    EV_O   [H, d_head, d_head]   eigenvectors of initial S_O
    lam_V  [H, d_head]           eigenvalues descending
    lam_O  [H, d_head]

Full weight reconstruction (vectorised — no Python loops over heads):
    pure:   S = EV @ diag_embed(lam) @ EV.mT      gradient via lam
    hybrid: S = S (direct)                          gradient via S

    W_V = bmm(Q_V, S_V) → [H, n_embd, d_head] → permute+reshape → [n_embd, n_embd]
    W_O = bmm(S_O, Q_O) → [H, d_head, n_embd] →         reshape → [n_embd, n_embd]

Conv1D convention (GPT-2 / HuggingFace)
────────────────────────────────────────
GPT-2 uses Conv1D where output = x @ weight + bias.
weight shape is [in, out] — the transpose of a standard nn.Linear weight.
    c_attn.weight  [n_embd, 3*n_embd]   QKV fused projection
    c_proj.weight  [n_embd,   n_embd]   output projection

Our reconstructed W_V and W_O match this convention exactly:
    W_V [n_embd, n_embd]  — V columns concatenated: [W_V_0 | W_V_1 | ...]
    W_O [n_embd, n_embd]  — O rows concatenated:    [W_O_0 ; W_O_1 ; ...]

CPU offloading
──────────────
PAFTModel.__init__ calls _decompose_all() which runs the full polar decomposition
on CPU.  The caller (BaseMethod.build) moves the model to device only after this
returns.  This keeps VRAM at zero during the O(n_layers x n_heads) SVD pass.

Gradient checkpointing
───────────────────────
gradient_checkpointing_enable() delegates to the underlying HuggingFace model,
which applies torch.utils.checkpoint to each transformer block.  Because
PAFTAttention is inserted into transformer.h[l].attn, its forward() — including
the weight reconstruction — is recomputed during the backward pass rather than
stored.  This trades ~30% extra compute for ~40% VRAM reduction on activations.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from paft.decomposition.polar import polar_decompose, polar_decompose_left
from paft.model.extractor import (
    get_gpt2_dims,
    extract_wv_head,
    extract_wo_head,
    extract_wq_wk_layer,
    extract_biases_layer,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PAFTAttention
# ──────────────────────────────────────────────────────────────────────────────

class PAFTAttention(nn.Module):
    """
    Decomposed attention layer that replaces GPT2Attention.

    Stores all polar decomposition components for all heads in a single layer.
    Mode ('pure' | 'hybrid') controls forward-pass reconstruction only — it does
    NOT affect what is stored.  This means the same model can switch between pure
    and hybrid without re-decomposing, and checkpoints are mode-agnostic.

    parameter_groups / method._configure_parameters() controls which parameters
    have requires_grad=True.  The forward handles both modes regardless.

    Hot-path design
    ───────────────
    _get_S_V / _get_S_O and reconstruct_W_V / reconstruct_W_O are called on
    every forward pass.  All are fully vectorised with torch.bmm and
    torch.diag_embed — zero Python loops over heads.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        Q_V:    torch.Tensor,   # [H, n_embd, d_head]  frozen semi-orthogonal
        Q_O:    torch.Tensor,   # [H, d_head, n_embd]  frozen semi-orthogonal
        EV_V:   torch.Tensor,   # [H, d_head, d_head]  frozen eigenvectors of S_V
        EV_O:   torch.Tensor,   # [H, d_head, d_head]  frozen eigenvectors of S_O
        S_V:    torch.Tensor,   # [H, d_head, d_head]  init scaling — hybrid trains
        S_O:    torch.Tensor,   # [H, d_head, d_head]
        lam_V:  torch.Tensor,   # [H, d_head]          init eigenvalues — pure trains
        lam_O:  torch.Tensor,   # [H, d_head]
        W_Q:    torch.Tensor,   # [n_embd, n_embd]     frozen Q projection (all heads)
        W_K:    torch.Tensor,   # [n_embd, n_embd]     frozen K projection (all heads)
        b_qkv:  torch.Tensor,   # [3*n_embd]           QKV biases — safe variants train
        b_o:    torch.Tensor,   # [n_embd]             output bias — safe variants train
        mode:   str = "hybrid",
    ) -> None:
        super().__init__()

        self.H         = config.n_head
        self.n_embd    = config.n_embd
        self.d_head    = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.scale     = 1.0 / math.sqrt(self.d_head)
        self.mode      = mode   # set by method after build; mutable at any time

        # ── frozen buffers ─────────────────────────────────────────────────
        # register_buffer: participates in .to(device), not in optimizer state.
        self.register_buffer("Q_V",  Q_V)
        self.register_buffer("Q_O",  Q_O)
        self.register_buffer("EV_V", EV_V)
        self.register_buffer("EV_O", EV_O)
        self.register_buffer("W_Q",  W_Q)
        self.register_buffer("W_K",  W_K)

        # Causal mask — lower-triangular bool, same semantics as GPT2Attention
        max_pos = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_pos, max_pos, dtype=torch.bool))
              .view(1, 1, max_pos, max_pos),
        )

        # ── trainable parameters (ALL start frozen) ────────────────────────
        # _configure_parameters() in the method class selectively unfreezes
        # whichever subset is appropriate for that method.
        self.S_V   = nn.Parameter(S_V.clone(),   requires_grad=False)
        self.S_O   = nn.Parameter(S_O.clone(),   requires_grad=False)
        self.lam_V = nn.Parameter(lam_V.clone(), requires_grad=False)
        self.lam_O = nn.Parameter(lam_O.clone(), requires_grad=False)
        self.b_qkv = nn.Parameter(b_qkv.clone(), requires_grad=False)
        self.b_o   = nn.Parameter(b_o.clone(),   requires_grad=False)

    # ── S reconstruction ──────────────────────────────────────────────────────

    def _get_S_V(self) -> torch.Tensor:
        """
        Return current [H, d_head, d_head] scaling matrix for V projection.

        pure:   S_V = EV_V @ diag_embed(lam_V) @ EV_V^T
                Gradient flows through lam_V only.
                EV_V is frozen — direction is fixed, only scaling magnitudes move.

        hybrid: S_V returned directly.
                Gradient flows through S_V (full d_head x d_head matrix per head).

        Vectorised: no Python loop.
            EV_V:            [H, d, d]
            diag_embed(lam): [H, d, d]
            matmul result:   [H, d, d]
        """
        if self.mode == "pure":
            return self.EV_V @ torch.diag_embed(self.lam_V) @ self.EV_V.mT
        return self.S_V

    def _get_S_O(self) -> torch.Tensor:
        """Return current [H, d_head, d_head] scaling matrix for O projection."""
        if self.mode == "pure":
            return self.EV_O @ torch.diag_embed(self.lam_O) @ self.EV_O.mT
        return self.S_O

    # ── weight reconstruction ────────────────────────────────────────────────

    def reconstruct_W_V(self) -> torch.Tensor:
        """
        Reconstruct full V-projection weight [n_embd, n_embd].

        W_V_h = Q_V[h] @ S_V[h]   (right polar)

        Vectorised:
            bmm(Q_V, S_V)             [H, n_embd, d_head]
            permute(1, 0, 2)          [n_embd, H, d_head]
            reshape(n_embd, n_embd)   columns interleaved by head — matches
                                      the original c_attn.weight V-slice layout.
        """
        S   = self._get_S_V()                                     # [H, d, d]
        W_h = torch.bmm(self.Q_V, S)                             # [H, n_embd, d_head]
        return W_h.permute(1, 0, 2).reshape(self.n_embd, self.n_embd)

    def reconstruct_W_O(self) -> torch.Tensor:
        """
        Reconstruct full O-projection weight [n_embd, n_embd].

        W_O_h = S_O[h] @ Q_O[h]   (left polar)

        Vectorised:
            bmm(S_O, Q_O)             [H, d_head, n_embd]
            reshape(n_embd, n_embd)   rows stacked by head — matches
                                      the original c_proj.weight layout.
        """
        S   = self._get_S_O()                                     # [H, d, d]
        W_h = torch.bmm(S, self.Q_O)                             # [H, d_head, n_embd]
        return W_h.reshape(self.n_embd, self.n_embd)

    # ── per-head accessors (geometric health snapshot, no-grad context) ───────

    def get_W_V_per_head(self) -> torch.Tensor:
        """Return [H, n_embd, d_head] — per-head W_V without concatenation."""
        return torch.bmm(self.Q_V, self._get_S_V())

    def get_W_O_per_head(self) -> torch.Tensor:
        """Return [H, d_head, n_embd] — per-head W_O without concatenation."""
        return torch.bmm(self._get_S_O(), self.Q_O)

    # ── head splitting / merging ─────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, n_embd] → [B, H, T, d_head]"""
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.d_head).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, d_head] → [B, T, n_embd]"""
        B, _, T, _ = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_embd)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states:     torch.Tensor,
        attention_mask:    Optional[torch.Tensor] = None,
        layer_past:        Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask:         Optional[torch.Tensor] = None,
        use_cache:         bool = False,
        output_attentions: bool = False,
    ) -> Tuple:
        """
        Forward compatible with GPT2Attention's output signature.
        Returns (attn_output [B, T, n_embd], present | None [, attn_weights]).

        Reconstruction happens here on every forward call.  With gradient
        checkpointing, reconstruction is recomputed during backward rather than
        stored — this is the desired behaviour (saves VRAM, trades compute).

        GPT-2 Conv1D projection convention (weight is [in, out]):
            q = hidden_states @ W_Q + b_q
            k = hidden_states @ W_K + b_k
            v = hidden_states @ W_V + b_v       ← gradient via S_V or lam_V
            output = context @ W_O + b_o        ← gradient via S_O or lam_O
        """
        # Reconstruct — gradient entry point for trainable S / lam
        W_V = self.reconstruct_W_V()   # [n_embd, n_embd]
        W_O = self.reconstruct_W_O()   # [n_embd, n_embd]

        B, T, n = hidden_states.shape

        # Bias slices — b_qkv stores [b_q | b_k | b_v] matching c_attn.bias layout
        b_q = self.b_qkv[:n]
        b_k = self.b_qkv[n : 2 * n]
        b_v = self.b_qkv[2 * n :]

        # QKV projections
        q = hidden_states @ self.W_Q + b_q   # [B, T, n_embd]  — frozen W_Q
        k = hidden_states @ self.W_K + b_k   # [B, T, n_embd]  — frozen W_K
        v = hidden_states @ W_V  + b_v       # [B, T, n_embd]  — trainable W_V

        q = self._split_heads(q)             # [B, H, T, d_head]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # KV cache
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None
        kv_len  = k.shape[2]
        q_len   = q.shape[2]

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,kv]

        # Causal mask — slice handles KV cache offset correctly
        mask = self.causal_mask[:, :, kv_len - q_len : kv_len, :kv_len]
        attn_weights = attn_weights.masked_fill(~mask, torch.finfo(attn_weights.dtype).min)

        if attention_mask is not None:
            # HF additive mask: [B, 1, 1, kv_len], 0 = keep, -inf = mask
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1).to(v.dtype)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context     = torch.matmul(attn_weights, v)   # [B, H, T, d_head]
        context     = self._merge_heads(context)       # [B, T, n_embd]
        attn_output = context @ W_O + self.b_o        # [B, T, n_embd] — trainable W_O

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# ──────────────────────────────────────────────────────────────────────────────
# PAFTModel
# ──────────────────────────────────────────────────────────────────────────────

class PAFTModel(nn.Module):
    """
    Wraps GPT2LMHeadModel, replacing every GPT2Attention with PAFTAttention.

    Construction (all on CPU, before .to(device)):
        1. Load the pretrained GPT2LMHeadModel (caller's responsibility).
        2. For each transformer layer: extract W_V/W_O, polar-decompose all heads,
           eigendecompose each S, build PAFTAttention, swap in-place.
        3. All PAFTAttention parameters start frozen (requires_grad=False).
        4. Caller's _configure_parameters() unfreezes the correct subset.

    The base model's original c_attn and c_proj weights become dead weight
    (no forward path through them) but remain in self.base.state_dict() for
    reference.  Storage cost: ~9 MB for GPT-2 small on CPU RAM.

    Gradient checkpointing:
        Delegated to self.base.gradient_checkpointing_enable().  HF applies
        torch.utils.checkpoint to each Block in transformer.h.  The block
        calls PAFTAttention.forward — so weight reconstruction is recomputed
        during backward, not stored, reducing activation VRAM.
    """

    def __init__(self, base_model: GPT2LMHeadModel) -> None:
        super().__init__()
        self.base      = base_model
        self._n_layers = base_model.config.n_layer
        self._decompose_all()

    # ── decomposition ─────────────────────────────────────────────────────────

    def _decompose_all(self) -> None:
        """Replace all GPT2Attention layers.  Runs entirely on CPU."""
        dims = get_gpt2_dims(self.base)
        logger.info(
            f"PAFTModel: decomposing {dims.n_layers} layers x "
            f"{dims.n_heads} heads on CPU ..."
        )
        for l in range(dims.n_layers):
            self.base.transformer.h[l].attn = self._build_paft_attention(l, dims)
            if (l + 1) % 4 == 0 or l == dims.n_layers - 1:
                logger.info(f"  layer {l + 1}/{dims.n_layers} done")

    def _build_paft_attention(self, layer_idx: int, dims: dict) -> PAFTAttention:
        """
        Extract, decompose, and eigendecompose all heads for one transformer layer.

        Per-head operations:
            W_V_h [n_embd, d_head] -- polar_decompose      --> (Q_V_h, S_V_h)
                                   -- eigh(S_V_h)          --> (lam_V_h, EV_V_h)
            W_O_h [d_head, n_embd] -- polar_decompose_left --> (S_O_h, Q_O_h)
                                   -- eigh(S_O_h)          --> (lam_O_h, EV_O_h)

        torch.linalg.eigh is used (not eig) because S is symmetric PSD by
        construction from polar decomposition.  eigh is faster, more stable,
        and guarantees real-valued eigenvalues.

        eigh returns eigenvalues ascending — we flip to descending so that
        lam[0] is the dominant eigenvalue, matching singular value convention.

        All tensors are fp32 on CPU.
        """
        H   = dims.n_heads
        cfg = self.base.config

        Q_V_l,  S_V_l,  EV_V_l,  lam_V_l  = [], [], [], []
        Q_O_l,  S_O_l,  EV_O_l,  lam_O_l  = [], [], [], []

        for h in range(H):
            # V-side  — right polar: W_V_h = Q_V_h @ S_V_h
            W_V_h        = extract_wv_head(self.base, layer_idx, h).float()
            Q_V_h, S_V_h = polar_decompose(W_V_h)            # [n,d], [d,d]
            lam_V_h, EV_V_h = torch.linalg.eigh(S_V_h)       # ascending
            lam_V_h = lam_V_h.flip(0)                         # → descending
            EV_V_h  = EV_V_h.flip(1)

            Q_V_l.append(Q_V_h);   S_V_l.append(S_V_h)
            EV_V_l.append(EV_V_h); lam_V_l.append(lam_V_h)

            # O-side  — left polar: W_O_h = S_O_h @ Q_O_h
            W_O_h         = extract_wo_head(self.base, layer_idx, h).float()
            S_O_h, Q_O_h  = polar_decompose_left(W_O_h)      # [d,d], [d,n]
            lam_O_h, EV_O_h = torch.linalg.eigh(S_O_h)
            lam_O_h = lam_O_h.flip(0)
            EV_O_h  = EV_O_h.flip(1)

            Q_O_l.append(Q_O_h);   S_O_l.append(S_O_h)
            EV_O_l.append(EV_O_h); lam_O_l.append(lam_O_h)

        W_Q, W_K   = extract_wq_wk_layer(self.base, layer_idx)
        b_qkv, b_o = extract_biases_layer(self.base, layer_idx)

        return PAFTAttention(
            config    = cfg,
            layer_idx = layer_idx,
            Q_V       = torch.stack(Q_V_l).float(),      # [H, n_embd, d_head]
            Q_O       = torch.stack(Q_O_l).float(),      # [H, d_head, n_embd]
            EV_V      = torch.stack(EV_V_l).float(),     # [H, d_head, d_head]
            EV_O      = torch.stack(EV_O_l).float(),
            S_V       = torch.stack(S_V_l).float(),      # [H, d_head, d_head]
            S_O       = torch.stack(S_O_l).float(),
            lam_V     = torch.stack(lam_V_l).float(),    # [H, d_head]
            lam_O     = torch.stack(lam_O_l).float(),
            W_Q       = W_Q.float(),                     # [n_embd, n_embd]
            W_K       = W_K.float(),
            b_qkv     = b_qkv.float(),                   # [3*n_embd]
            b_o       = b_o.float(),                     # [n_embd]
        )

    # ── mode control ──────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """
        Set reconstruction mode on all PAFTAttention layers.
        'pure' → lam_V/lam_O used in forward.
        'hybrid' → S_V/S_O used in forward.
        Called by method._build_model(), before _configure_parameters().
        """
        if mode not in ("pure", "hybrid"):
            raise ValueError(f"mode must be 'pure' or 'hybrid', got {mode!r}")
        for _, attn in self.iter_paft_attentions():
            attn.mode = mode

    # ── accessors ─────────────────────────────────────────────────────────────

    def iter_paft_attentions(self):
        """
        Yield (layer_idx, PAFTAttention) for every transformer layer.
        Raises RuntimeError if a layer was not replaced (defensive check).
        Used by method files in _configure_parameters() and paft_snapshot().
        """
        for l in range(self._n_layers):
            attn = self.base.transformer.h[l].attn
            if not isinstance(attn, PAFTAttention):
                raise RuntimeError(
                    f"Layer {l} attention is {type(attn).__name__}, "
                    "expected PAFTAttention — was _decompose_all() called?"
                )
            yield l, attn

    def get_paft_attention(self, layer_idx: int) -> PAFTAttention:
        """Direct access to one layer's PAFTAttention.  Unchecked."""
        return self.base.transformer.h[layer_idx].attn

    # ── forward / HuggingFace passthrough ─────────────────────────────────────

    def forward(self, *args, **kwargs):
        """
        Delegate to GPT2LMHeadModel.  It calls each block, which calls
        PAFTAttention.forward, which reconstructs W_V / W_O transparently.
        """
        return self.base(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """
        Delegate to HF model.  HF wraps each transformer Block with
        torch.utils.checkpoint, causing PAFTAttention.forward (including
        weight reconstruction) to be recomputed on backward.
        """
        self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        self.base.gradient_checkpointing_disable(**kwargs)

    @property
    def config(self):
        return self.base.config

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        """
        Saves the underlying HF model (with dead c_attn/c_proj weights).
        For full PAFT checkpointing use CheckpointSaver, which saves the
        PAFTModel state_dict including S_V, lam_V, etc.
        """
        return self.base.save_pretrained(*args, **kwargs)
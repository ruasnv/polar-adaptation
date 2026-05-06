"""
SVFAttention and SVFModel — Singular Value Fine-tuning.

SVF is structurally the closest method to pure_paft:
    SVF:       freeze U, Vh from SVD;  train sigma (diagonal singular values)
    pure_paft: freeze Q, EV from polar; train lambda (eigenvalues of S)

Key differences that the paper isolates via this comparison:
    1. Factorisation: SVD (W = U @ diag(sigma) @ Vh) vs
                      polar (W = Q @ S, S = EV @ diag(lam) @ EV^T)
    2. Basis meaning: SVD right singular vectors (Vh) capture energy-sorted input
                      directions; polar eigenvectors (EV) capture the symmetric
                      scaling directions of the OV circuit's transformation.
    3. Symmetry:      SVD sigma is always non-negative; polar lam may be negative
                      for indefinite S (though S is PSD at init from polar decomp).

Identical parameter count to pure_paft for fair comparison:
    12 layers × 12 heads × 2 sides × 64 = 18,432 (GPT-2 small)

Shapes:
    W_V_h [n_embd, d_head] — economy SVD → U_V [n_embd, d_head], sigma_V [d_head], Vh_V [d_head, d_head]
    W_O_h [d_head, n_embd] — economy SVD → U_O [d_head, d_head], sigma_O [d_head], Vh_O [d_head, n_embd]

Reconstruction (vectorised):
    W_V = bmm(bmm(U_V, diag_embed(sigma_V)), Vh_V) → [H, n_embd, d_head] → permute+reshape
    W_O = bmm(bmm(U_O, diag_embed(sigma_O)), Vh_O) → [H, d_head, n_embd] → reshape
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from paft.model.extractor import get_gpt2_dims, extract_layer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SVFAttention
# ──────────────────────────────────────────────────────────────────────────────

class SVFAttention(nn.Module):
    """
    Replaces GPT2Attention.  Stores U, sigma, Vh from economy SVD of each head's
    W_V and W_O.  Only sigma_V and sigma_O are trainable.

    Forward reconstruction is vectorised — no Python loop over heads.
    Gradient flows through sigma only; U and Vh are frozen buffers.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        U_V:     torch.Tensor,   # [H, n_embd, d_head]  frozen left  singular vectors of W_V
        sigma_V: torch.Tensor,   # [H, d_head]           singular values of W_V — trainable
        Vh_V:    torch.Tensor,   # [H, d_head, d_head]  frozen right singular vectors of W_V
        U_O:     torch.Tensor,   # [H, d_head, d_head]  frozen left  singular vectors of W_O
        sigma_O: torch.Tensor,   # [H, d_head]           singular values of W_O — trainable
        Vh_O:    torch.Tensor,   # [H, d_head, n_embd]  frozen right singular vectors of W_O
        W_Q:     torch.Tensor,   # [n_embd, n_embd]     frozen
        W_K:     torch.Tensor,   # [n_embd, n_embd]     frozen
        b_qkv:   torch.Tensor,   # [3*n_embd]           frozen
        b_o:     torch.Tensor,   # [n_embd]             frozen
    ) -> None:
        super().__init__()

        self.H         = config.n_head
        self.n_embd    = config.n_embd
        self.d_head    = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.scale     = 1.0 / math.sqrt(self.d_head)

        # Frozen buffers
        self.register_buffer("U_V",  U_V)
        self.register_buffer("Vh_V", Vh_V)
        self.register_buffer("U_O",  U_O)
        self.register_buffer("Vh_O", Vh_O)
        self.register_buffer("W_Q",  W_Q)
        self.register_buffer("W_K",  W_K)

        max_pos = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_pos, max_pos, dtype=torch.bool))
              .view(1, 1, max_pos, max_pos),
        )

        # Trainable parameters — both start frozen; svf.py unfreezes sigma_V/O
        self.sigma_V = nn.Parameter(sigma_V.clone(), requires_grad=False)
        self.sigma_O = nn.Parameter(sigma_O.clone(), requires_grad=False)
        self.b_qkv   = nn.Parameter(b_qkv.clone(),  requires_grad=False)
        self.b_o     = nn.Parameter(b_o.clone(),    requires_grad=False)

    # ── reconstruction ────────────────────────────────────────────────────────

    def reconstruct_W_V(self) -> torch.Tensor:
        """
        W_V = U_V @ diag_embed(sigma_V) @ Vh_V   →  [n_embd, n_embd]

        Vectorised:
            bmm(U_V [H,n,d], diag_embed(sigma_V) [H,d,d]) = [H,n,d]
            bmm(result,      Vh_V                [H,d,d]) = [H,n,d]
            permute(1,0,2) → [n,H,d] → reshape → [n_embd, n_embd]
        """
        tmp = torch.bmm(self.U_V, torch.diag_embed(self.sigma_V))  # [H, n_embd, d_head]
        W_h = torch.bmm(tmp, self.Vh_V)                            # [H, n_embd, d_head]
        return W_h.permute(1, 0, 2).reshape(self.n_embd, self.n_embd)

    def reconstruct_W_O(self) -> torch.Tensor:
        """
        W_O = U_O @ diag_embed(sigma_O) @ Vh_O   →  [n_embd, n_embd]

        Vectorised:
            bmm(U_O [H,d,d], diag_embed(sigma_O) [H,d,d]) = [H,d,d]
            bmm(result,      Vh_O                [H,d,n]) = [H,d,n]
            reshape → [n_embd, n_embd]
        """
        tmp = torch.bmm(self.U_O, torch.diag_embed(self.sigma_O))  # [H, d_head, d_head]
        W_h = torch.bmm(tmp, self.Vh_O)                            # [H, d_head, n_embd]
        return W_h.reshape(self.n_embd, self.n_embd)

    def get_W_V_per_head(self) -> torch.Tensor:
        """[H, n_embd, d_head] — for geometric health snapshot."""
        tmp = torch.bmm(self.U_V, torch.diag_embed(self.sigma_V))
        return torch.bmm(tmp, self.Vh_V)

    def get_W_O_per_head(self) -> torch.Tensor:
        """[H, d_head, n_embd] — for geometric health snapshot."""
        tmp = torch.bmm(self.U_O, torch.diag_embed(self.sigma_O))
        return torch.bmm(tmp, self.Vh_O)

    # ── head utils ────────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.d_head).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_embd)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states:     torch.Tensor,
        attention_mask:    Optional[torch.Tensor] = None,
        layer_past:        Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask:         Optional[torch.Tensor] = None,
        use_cache:         bool = False,
        output_attentions: bool = False,
        past_key_values:   Optional[Any] = None,   # new-style HF cache (≥4.38)
        **kwargs,                                   # absorb any other new HF args
    ) -> Tuple:
        # Normalise cache argument
        if layer_past is None and past_key_values is not None:
            if isinstance(past_key_values, tuple):
                layer_past = past_key_values
        W_V = self.reconstruct_W_V()
        W_O = self.reconstruct_W_O()

        B, T, n = hidden_states.shape
        b_q = self.b_qkv[:n];  b_k = self.b_qkv[n:2*n];  b_v = self.b_qkv[2*n:]

        q = hidden_states @ self.W_Q + b_q
        k = hidden_states @ self.W_K + b_k
        v = hidden_states @ W_V  + b_v

        q = self._split_heads(q);  k = self._split_heads(k);  v = self._split_heads(v)

        if layer_past is not None:
            k = torch.cat([layer_past[0], k], dim=2)
            v = torch.cat([layer_past[1], v], dim=2)

        present = (k, v) if use_cache else None
        kv_len, q_len = k.shape[2], q.shape[2]

        attn_w = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask   = self.causal_mask[:, :, kv_len - q_len : kv_len, :kv_len]
        attn_w = attn_w.masked_fill(~mask, torch.finfo(attn_w.dtype).min)
        if attention_mask is not None:
            attn_w = attn_w + attention_mask
        attn_w = F.softmax(attn_w, dim=-1).to(v.dtype)
        if head_mask is not None:
            attn_w = attn_w * head_mask

        context     = self._merge_heads(torch.matmul(attn_w, v))
        attn_output = context @ W_O + self.b_o

        out = (attn_output, present)
        if output_attentions:
            out += (attn_w,)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# SVFModel
# ──────────────────────────────────────────────────────────────────────────────

class SVFModel(nn.Module):
    """
    Wraps GPT2LMHeadModel, replacing every GPT2Attention with SVFAttention.

    Decomposition (on CPU, before .to(device)):
        For each (layer, head): economy SVD of W_V_h and W_O_h.
        sigma is initialised to the pretrained singular values.
        U and Vh are frozen buffers.
    """

    def __init__(self, base_model: GPT2LMHeadModel) -> None:
        super().__init__()
        self.base      = base_model
        self._n_layers = base_model.config.n_layer
        self._decompose_all()

    def _decompose_all(self) -> None:
        """Two-pass: extract all layers first, then replace — safe against
        extraction-after-surgery bugs."""
        dims = get_gpt2_dims(self.base)
        logger.info(
            f"SVFModel: decomposing {dims.n_layers} layers x {dims.n_heads} heads on CPU ..."
        )
        # Pass 1: extract everything while all layers are still GPT2Attention
        all_lw = [extract_layer(self.base, l, dims) for l in range(dims.n_layers)]

        # Pass 2: build SVFAttention from extracted weights, replace in-place
        for l, lw in enumerate(all_lw):
            self.base.transformer.h[l].attn = self._build_svf_attention(l, lw, dims)
            if (l + 1) % 4 == 0 or l == dims.n_layers - 1:
                logger.info(f"  layer {l + 1}/{dims.n_layers} done")

    def _build_svf_attention(self, layer_idx: int, lw, dims) -> SVFAttention:
        H, n, d = dims.n_heads, dims.n_embd, dims.d_head

        U_V_l,  sigma_V_l,  Vh_V_l  = [], [], []
        U_O_l,  sigma_O_l,  Vh_O_l  = [], [], []

        for h in range(H):
            # W_V_h [n_embd, d_head] — tall matrix, economy SVD
            W_V_h           = lw.W_V[h].float()       # [n_embd, d_head]
            U_h, s_h, Vh_h  = torch.linalg.svd(W_V_h, full_matrices=False)
            # U_h [n, d], s_h [d], Vh_h [d, d]
            U_V_l.append(U_h);  sigma_V_l.append(s_h);  Vh_V_l.append(Vh_h)

            # W_O_h [d_head, n_embd] — wide matrix, economy SVD
            W_O_h           = lw.W_O[h].float()       # [d_head, n_embd]
            U_h, s_h, Vh_h  = torch.linalg.svd(W_O_h, full_matrices=False)
            # U_h [d, d], s_h [d], Vh_h [d, n]
            U_O_l.append(U_h);  sigma_O_l.append(s_h);  Vh_O_l.append(Vh_h)

        return SVFAttention(
            config   = self.base.config,
            layer_idx= layer_idx,
            U_V      = torch.stack(U_V_l).float(),      # [H, n_embd, d_head]
            sigma_V  = torch.stack(sigma_V_l).float(),  # [H, d_head]
            Vh_V     = torch.stack(Vh_V_l).float(),     # [H, d_head, d_head]
            U_O      = torch.stack(U_O_l).float(),      # [H, d_head, d_head]
            sigma_O  = torch.stack(sigma_O_l).float(),  # [H, d_head]
            Vh_O     = torch.stack(Vh_O_l).float(),     # [H, d_head, n_embd]
            W_Q      = lw.W_Q.float(),
            W_K      = lw.W_K.float(),
            b_qkv    = lw.b_qkv.float(),
            b_o      = lw.b_o.float(),
        )

    def iter_svf_attentions(self):
        """Yield (layer_idx, SVFAttention) for all layers."""
        for l in range(self._n_layers):
            attn = self.base.transformer.h[l].attn
            if not isinstance(attn, SVFAttention):
                raise RuntimeError(
                    f"Layer {l} is {type(attn).__name__}, expected SVFAttention"
                )
            yield l, attn

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        self.base.gradient_checkpointing_disable(**kwargs)

    @property
    def config(self):
        return self.base.config
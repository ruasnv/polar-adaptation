"""
PAFTLinear — polar-decomposed nn.Linear drop-in replacement.

Supports per-head decomposition in two modes:

  'row'  — decomposes weight by ROW-CHUNKS (value projection).
           weight [H*d, n_in] → per-head W_h = weight[h*d:(h+1)*d, :] [d, n_in] wide.
           Left polar:   W_h = S_h @ Q_h   →  Q_h [d, n_in],  S_h [d, d]
           Reconstruct:  cat(S_h @ Q_h, dim=0) → [H*d, n_in]

  'col'  — decomposes weight by COLUMN-CHUNKS (output projection).
           weight [n_out, H*d] → per-head W_h = weight[:, h*d:(h+1)*d] [n_out, d] tall.
           Right polar:  W_h = Q_h @ S_h   →  Q_h [n_out, d],  S_h [d, d]
           Reconstruct:  cat(Q_h @ S_h, dim=1) → [n_out, H*d]

Training modes:
  'hybrid' — S_h is a free nn.Parameter [H, d, d].  Full d×d gradient per head.
  'pure'   — S_h = EV_h @ diag(lam_h) @ EV_h^T.  Only lam [H, d] is trained.
             EV_h is frozen (initial eigenvectors of S_h from polar decomp).

Q is always frozen (registered as buffer, not parameter).
Bias (if any) is also frozen — not adapted by PAFT.

Memory layout for LLaMA-3.2-3B v_proj (8 KV heads, head_dim=128):
  Q [8, 128, 3072]  fp16 buffer  → 6.3 MB per layer
  S [8, 128, 128]   fp32 param   → 0.52 MB per layer
  28 layers total:  Q ~175 MB,  S ~15 MB  ✓ fits in 8 GB VRAM

Memory layout for DeBERTa-v3-base (12 heads, head_dim=64):
  Q [12, 64, 768]   fp16 buffer  → 0.6 MB per layer
  S [12, 64, 64]    fp32 param   → 0.19 MB per layer
  12 layers × 2 projections:  Q ~15 MB,  S ~4.6 MB  ✓
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple

from paft.decomposition.polar import polar_decompose, polar_decompose_left


class PAFTLinear(nn.Module):
    """
    PAFT-parameterised drop-in replacement for nn.Linear.

    After construction, every parameter starts frozen (requires_grad=False).
    The method class calls .lam.requires_grad_(True)  for pure mode, or
    .S.requires_grad_(True) for hybrid mode.

    Args:
        weight:     Pretrained weight [out_features, in_features].  Detached fp32.
        bias:       Optional bias [out_features].  Stored as frozen buffer.
        n_heads:    Number of heads to decompose over.
        head_dim:   Dimension per head (d).
        decomp_mode: 'row' for value proj, 'col' for output proj.
        train_mode:  'hybrid' (train full S) or 'pure' (train only eigenvalues).
        q_dtype:    Dtype for the frozen Q buffer.  fp16 recommended for LLaMA
                    to save memory; fp32 for DeBERTa where everything is fp32.
    """

    def __init__(
        self,
        weight:       torch.Tensor,
        bias:         Optional[torch.Tensor],
        n_heads:      int,
        head_dim:     int,
        decomp_mode:  Literal['row', 'col'] = 'row',
        train_mode:   Literal['hybrid', 'pure'] = 'hybrid',
        q_dtype:      torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_heads    = n_heads
        self.head_dim   = head_dim
        self.decomp_mode = decomp_mode
        self.train_mode  = train_mode

        # Frozen bias — not adapted
        if bias is not None:
            self.register_buffer('bias', bias.detach().clone().float())
        else:
            self.bias = None

        # Decompose per head
        Q, S, EV, lam = self._decompose(weight.detach().float(), n_heads, head_dim, decomp_mode)

        # Q: frozen isometric factor (cast to q_dtype to save memory if fp16)
        self.register_buffer('Q', Q.to(q_dtype))

        if train_mode == 'pure':
            # EV: frozen initial eigenvectors
            self.register_buffer('EV', EV.float())
            # lam: trainable eigenvalues [H, d] — starts frozen; method unfreezes
            self.lam = nn.Parameter(lam.float(), requires_grad=False)
        else:
            # S: trainable full scaling matrix [H, d, d]
            self.S = nn.Parameter(S.float(), requires_grad=False)

        # Store initial S as frozen buffer for geometric analysis
        self.register_buffer('S_init', S.float())

    # ── decomposition ─────────────────────────────────────────────────────────

    @staticmethod
    def _decompose(
        weight: torch.Tensor,
        n_heads: int,
        head_dim: int,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose weight per head.  Returns (Q, S, EV, lam) all in fp32.

        Row mode: weight [H*d, n_in]  → per head [d, n_in] wide → left polar
        Col mode: weight [n_out, H*d] → per head [n_out, d] tall → right polar
        """
        Q_list, S_list, EV_list, lam_list = [], [], [], []

        for h in range(n_heads):
            if mode == 'row':
                W_h = weight[h * head_dim:(h + 1) * head_dim, :]  # [d, n_in] wide
                S_h, Q_h = polar_decompose_left(W_h)               # [d,d], [d,n_in]
            else:
                W_h = weight[:, h * head_dim:(h + 1) * head_dim]   # [n_out, d] tall
                Q_h, S_h = polar_decompose(W_h)                     # [n_out,d], [d,d]

            # Eigendecompose S for pure mode
            lam_h, EV_h = torch.linalg.eigh(S_h)  # ascending eigenvalues
            lam_h = lam_h.flip(0)                  # descending
            EV_h  = EV_h.flip(1)                   # matching eigenvectors

            Q_list.append(Q_h)
            S_list.append(S_h)
            EV_list.append(EV_h)
            lam_list.append(lam_h)

        return (
            torch.stack(Q_list),    # Q   [H, d, n_in] or [H, n_out, d]
            torch.stack(S_list),    # S   [H, d, d]
            torch.stack(EV_list),   # EV  [H, d, d]
            torch.stack(lam_list),  # lam [H, d]
        )

    # ── S reconstruction ──────────────────────────────────────────────────────

    def _get_S(self) -> torch.Tensor:
        """Return current [H, d, d] scaling matrix."""
        if self.train_mode == 'pure':
            # EV [H, d, d],  lam [H, d]
            lam_diag = torch.diag_embed(self.lam)                         # [H, d, d]
            return self.EV @ lam_diag @ self.EV.transpose(-1, -2)         # [H, d, d]
        return self.S

    # ── weight reconstruction ─────────────────────────────────────────────────

    def reconstruct_weight(self) -> torch.Tensor:
        """
        Reconstruct the full weight matrix [out_features, in_features].
        Called on every forward pass — kept vectorised, no Python loops.
        """
        S = self._get_S()  # [H, d, d] (Float)
        Q = self.Q.to(dtype=S.dtype)  # Cast Q to Float for high-precision arithmetic

        if self.decomp_mode == 'row':
            # W_h = S_h @ Q_h  →  [H, d, n_in]  →  reshape to [H*d, n_in]
            W_h = torch.bmm(S, Q)                              # [H, d, n_in]
            H, d, n_in = W_h.shape
            return W_h.reshape(H * d, n_in)
        else:
            # W_h = Q_h @ S_h  →  [H, n_out, d]  →  permute+reshape to [n_out, H*d]
            W_h = torch.bmm(Q, S)                              # [H, n_out, d]
            H, n_out, d = W_h.shape
            # Each W_h[:, h, :] fills columns h*d:(h+1)*d of the output
            return W_h.permute(1, 0, 2).reshape(n_out, H * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.reconstruct_weight()

        # FIXED: Match weight and bias precision to input (e.g., Half for Colab T4)
        W = W.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        return F.linear(x, W, bias)

    # ── geometric accessors ───────────────────────────────────────────────────

    def get_Q_S(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Q [H,d,n or H,n,d], S [H,d,d]) on CPU.  No-grad context."""
        with torch.no_grad():
            return self.Q.cpu(), self._get_S().detach().cpu()

    def orthogonality_error(self) -> float:
        """
        Mean ||Q_h^T Q_h - I||_F or ||Q_h Q_h^T - I||_F across heads.
        Should be ~0 for row mode (semi-unitary rows), ~0 for col mode (semi-unitary cols).
        Used to measure Q orthogonality degradation from fp16 storage.
        """
        Q = self.Q.float()
        H = Q.shape[0]
        errs = []
        for h in range(H):
            Q_h = Q[h]
            if self.decomp_mode == 'row':
                # Q_h [d, n_in], expect Q_h @ Q_h^T ≈ I_d
                err = (Q_h @ Q_h.T - torch.eye(Q_h.shape[0], device=Q_h.device)).norm().item()
            else:
                # Q_h [n_out, d], expect Q_h^T @ Q_h ≈ I_d
                err = (Q_h.T @ Q_h - torch.eye(Q_h.shape[1], device=Q_h.device)).norm().item()
            errs.append(err)
        return sum(errs) / len(errs)
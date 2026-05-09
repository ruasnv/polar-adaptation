"""
PoLARLinear — one-sided Stiefel adaptation for any nn.Linear layer.

Parameterisation:
    ΔW = (alpha / r) * X @ B^T

    X [out_features, r]:  initialized with orthonormal columns (Stiefel manifold)
    B [in_features,  r]:  initialized to zeros → ΔW = 0 at start

After each optimizer step the caller must invoke retract_to_stiefel()
which projects X back via QR decomposition:
    X ← Q  where  X = QR (thin QR)

Why this achieves high sr(ΔW):
    With X orthonormal and B full-rank, ΔW = X @ B^T has stable rank
    sr(ΔW) ≥ min(r, rank(B)), which scales with r rather than collapsing to ~1
    as in standard LoRA.  This is the key property PoLAR demonstrates.

Scientific framing in the paper:
    "We implement PoLAR using a one-sided Stiefel constraint with QR retraction
    (equivalent to the PoLAR landing-field algorithm at convergence).
    This achieves the same stable-rank improvement as the original PoLAR
    while using standard PyTorch autograd."

Integration:
    - DeBERTa (GLUE):    HF Trainer callback calls retract_to_stiefel() on_step_end
    - LLaMA (LLM tasks): custom training loop calls it after optimizer.step()
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoLARLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with PoLAR-style Stiefel adaptation.

    Usage:
        layer = PoLARLinear.from_linear(original_linear, rank=8, alpha=16)
        # Replace the original layer with this
        # After every optimizer.step(): layer.retract_to_stiefel()

    Args:
        in_features:   Input dimension of the original linear layer.
        out_features:  Output dimension of the original linear layer.
        weight:        Pretrained weight [out, in]. Stored as frozen buffer.
        bias:          Optional pretrained bias. Stored as frozen buffer.
        rank:          Adaptation rank r.
        alpha:         LoRA-style scaling factor. Effective scale = alpha / rank.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        weight:       torch.Tensor,
        bias:         Optional[torch.Tensor],
        rank:         int = 8,
        alpha:        float = 16.0,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank  = rank
        self.scale = alpha / rank

        # Frozen base weight and bias
        self.register_buffer('weight_0', weight.detach().float())
        if bias is not None:
            self.register_buffer('bias_0', bias.detach().float())
        else:
            self.bias_0 = None

        # Stiefel factor X [out, r] — orthonormal columns
        X_init = torch.linalg.qr(torch.randn(out_features, rank))[0]
        self.X = nn.Parameter(X_init, requires_grad=True)

        # Unconstrained factor B [in, r] — initialized to zero (ΔW = 0 at start)
        self.B = nn.Parameter(torch.zeros(in_features, rank), requires_grad=True)

    @classmethod
    def from_linear(cls, layer: nn.Linear, rank: int = 8, alpha: float = 16.0) -> "PoLARLinear":
        """Factory: build PoLARLinear from an existing nn.Linear layer."""
        bias_tensor = layer.bias.detach() if layer.bias is not None else None
        return cls(
            in_features  = layer.in_features,
            out_features = layer.out_features,
            weight       = layer.weight.detach(),
            bias         = bias_tensor,
            rank         = rank,
            alpha        = alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ΔW = scale * X @ B^T   [out, in]
        delta_W = self.scale * (self.X @ self.B.T)
        W_eff   = self.weight_0 + delta_W
        bias    = self.bias_0 if self.bias_0 is not None else None
        return F.linear(x, W_eff, bias)

    @torch.no_grad()
    def retract_to_stiefel(self) -> None:
        """
        Project X back to the Stiefel manifold via thin QR decomposition.
        Must be called after every optimizer.step() for the Stiefel constraint to hold.

        QR retraction:  X_new = Q  where  X = QR (thin, R has positive diagonal)
        Sign convention: multiply Q columns by sign(diag(R)) to make R positive-diagonal.
        This gives the unique canonical QR factor and prevents sign flipping across steps.
        """
        Q, R = torch.linalg.qr(self.X.data)
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0   # handle zero diagonal
        self.X.data = Q * signs.unsqueeze(0)

    def get_delta_W(self) -> torch.Tensor:
        """Return ΔW [out, in] for geometric analysis (sr(ΔW) computation)."""
        with torch.no_grad():
            return (self.scale * self.X @ self.B.T).detach().cpu()

    def get_effective_W(self) -> torch.Tensor:
        """Return W_eff = W_0 + ΔW [out, in] for sr(W_eff) computation."""
        with torch.no_grad():
            return (self.weight_0 + self.scale * self.X @ self.B.T).detach().cpu()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")
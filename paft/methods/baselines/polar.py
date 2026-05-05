"""
PoLAR (M7) — Polar-decomposed Low-rank Adapter Representation.

Reference: "PoLAR: Polar-decomposed Low-rank Adapter Representation" (2024).

Parameterisation:
    ΔW = X @ Theta @ Y^T
    X ∈ St(m, r)  — left direction matrix, column-orthogonal (Stiefel manifold)
    Y ∈ St(n, r)  — right direction matrix, column-orthogonal (Stiefel manifold)
    Theta ∈ R^{r×r} — unconstrained scale matrix (unlike AdaLoRA's diagonal)
    W_new = W_0 + ΔW   (additive, W_0 frozen)

Landing field optimiser (Algorithm 2 from the paper):
    The Stiefel constraint is enforced via an infeasible method that avoids
    expensive SVD-based retraction.  Instead, gradient hooks modify the
    gradient of X and Y before AdamW sees them:

        grad_modified(Z) = ψ(Z) @ Z  +  λ * 4 * Z @ (Z^T @ Z - I_r)

    Where:
        ψ(Z) = Skew(grad_Z L @ Z^T) = (grad_Z L @ Z^T - Z @ grad_Z L^T) / 2
        First term:  Riemannian gradient — moves Z along the Stiefel manifold
        Second term: infeasibility penalty — pulls Z toward the manifold (Z^T Z → I)

    The two terms are orthogonal, decoupling loss minimisation from feasibility.
    Computation uses only matrix multiplications — no SVD, no retraction,
    GPU-friendly.

    Hooks are registered during _configure_parameters() via register_hook()
    on the X and Y Parameter tensors.  They fire during backward() and modify
    .grad in-place before AdamW reads it.  AdamW then applies its moment
    estimates to the already-corrected Riemannian gradient.

Why include as a baseline:
    PoLAR shares polar decomposition as motivation with PAFT, but is still
    ADDITIVE (W_new = W_0 + ΔW).  The comparison directly tests whether the
    Stiefel constraint is sufficient to preserve geometric health, or whether
    non-additivity (PAFT) is the essential property.

    Expected result: PoLAR improves over LoRA (Stiefel constraint helps)
    but still degrades geometric health relative to PAFT (additivity hurts).

Target modules: c_attn [n_embd, 3*n_embd] and c_proj [n_embd, n_embd].
Same scope as LoRA — only the attention projections are adapted.

Parameter count (r=8, GPT-2 small, n_embd=768, n_layers=12):
    c_attn: X [768, 8] + Theta [8, 8] + Y [2304, 8] = 6144 + 64 + 18432 = 24,640
    c_proj: X [768, 8] + Theta [8, 8] + Y [ 768, 8] = 6144 + 64 +  6144 = 12,352
    Per layer: 24,640 + 12,352 = 36,992
    Total: 12 * 36,992 = 443,904 ≈ 444K
    (comparable to lora_r8 at 442K — matched parameter bracket)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, freeze_all
from paft.model.extractor import get_gpt2_dims

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PolarAdapter — one ΔW = X @ Theta @ Y^T component
# ──────────────────────────────────────────────────────────────────────────────

class PolarAdapter(nn.Module):
    """
    Stores (X, Theta, Y) for one Conv1D layer's PoLAR adapter.

    X ∈ St(m, r),  Y ∈ St(n, r),  Theta ∈ R^{r×r}.

    All three start with requires_grad=False.
    PolarBaseline._configure_parameters() unfreezes them and registers
    the landing field gradient hooks on X and Y.

    Theta is initialised to zero so ΔW = 0 at the start of training.
    X and Y are initialised uniformly on the Stiefel manifold via QR
    decomposition of a random Gaussian matrix.
    """

    def __init__(self, m: int, n: int, rank: int, landing_lambda: float = 0.1) -> None:
        super().__init__()
        self.rank           = rank
        self.landing_lambda = landing_lambda

        # Stiefel initialisation: QR decomposition of a random Gaussian matrix
        # guarantees X^T X = I_r exactly at init.
        X_init, _ = torch.linalg.qr(torch.randn(m, rank))   # [m, r]
        Y_init, _ = torch.linalg.qr(torch.randn(n, rank))   # [n, r]

        self.X     = nn.Parameter(X_init,              requires_grad=False)  # [m, r]
        self.Theta = nn.Parameter(torch.zeros(rank, rank), requires_grad=False)  # [r, r]
        self.Y     = nn.Parameter(Y_init,              requires_grad=False)  # [n, r]

    def delta_weight(self) -> torch.Tensor:
        """
        Compute ΔW = X @ Theta @ Y^T.
        Gradient flows through all three factors.
        Shape: [m, n].
        """
        return self.X @ self.Theta @ self.Y.T

    def register_landing_hooks(self, landing_lambda: Optional[float] = None) -> None:
        """
        Register gradient hooks on X and Y that implement the landing field.

        Must be called AFTER X.requires_grad_(True) and Y.requires_grad_(True)
        — PyTorch requires requires_grad=True before register_hook.

        The hook modifies the gradient of Z (X or Y) from the raw autograd
        gradient ∇_Z L to the landing field Γ(Z):

            Γ(Z) = Skew(∇_Z L @ Z^T) @ Z  +  λ * 4 * Z @ (Z^T @ Z - I_r)

        The return value of the hook replaces .grad, which AdamW then reads.
        """
        lam = landing_lambda if landing_lambda is not None else self.landing_lambda
        self.X.register_hook(self._make_landing_hook(self.X, lam))
        self.Y.register_hook(self._make_landing_hook(self.Y, lam))

    @staticmethod
    def _make_landing_hook(Z_param: nn.Parameter, lam: float):
        """
        Factory — returns a hook closure over Z_param and lam.

        Using a factory (rather than a lambda) ensures each hook gets its own
        lam binding and the correct Z_param reference.

        Hook receives: grad [m, r] = ∇_Z L from autograd
        Hook returns:  modified_grad [m, r] = Γ(Z)
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            # Z_param.data: current value of Z, detached from computation graph.
            # Using .data avoids building a higher-order gradient graph.
            Z = Z_param.data   # [m, r]

            # ── Riemannian gradient ─────────────────────────────────────────
            # ψ(Z) = Skew(grad @ Z^T) = (grad @ Z^T - Z @ grad^T) / 2
            # Γ_riem = ψ(Z) @ Z
            GZt      = grad @ Z.T                    # [m, m]
            skew     = (GZt - GZt.T) * 0.5          # [m, m]  Skew-symmetric
            riem     = skew @ Z                      # [m, r]

            # ── Infeasibility penalty ───────────────────────────────────────
            # ∇N(Z) = 4 * Z @ (Z^T @ Z - I_r)   where N(Z) = ||Z^T Z - I||_F^2
            # Γ_penalty = λ * ∇N(Z)
            r        = Z.shape[1]
            I_r      = torch.eye(r, device=Z.device, dtype=Z.dtype)
            ZtZ_I    = Z.T @ Z - I_r               # [r, r]  deviation from identity
            penalty  = (lam * 4.0) * (Z @ ZtZ_I)  # [m, r]

            return riem + penalty

        return hook


# ──────────────────────────────────────────────────────────────────────────────
# PolarModel — wraps GPT2LMHeadModel with per-layer PolarAdapters
# ──────────────────────────────────────────────────────────────────────────────

class PolarModel(nn.Module):
    """
    Wraps GPT2LMHeadModel.  For each transformer layer, creates two
    PolarAdapters (one for c_attn, one for c_proj) and injects ΔW
    into the Conv1D forward via output hooks.

    The base model weights are never modified — only ΔW is added.

    Forward hook convention for Conv1D:
        Conv1D computes: output = input @ weight + bias
        Hook adds:       output + input @ ΔW(adapter)
        Net:             output = input @ (W + ΔW)  ✓ (additive PoLAR)
    """

    def __init__(
        self,
        base_model: GPT2LMHeadModel,
        rank:           int   = 8,
        landing_lambda: float = 0.1,
    ) -> None:
        super().__init__()
        self.base           = base_model
        self.rank           = rank
        self.landing_lambda = landing_lambda
        self._n_layers      = base_model.config.n_layer

        cfg    = base_model.config
        n_embd = cfg.n_embd

        # Build adapters and register forward hooks
        # adapters_c_attn[l]: PolarAdapter for layer l's c_attn [n_embd, 3*n_embd]
        # adapters_c_proj[l]: PolarAdapter for layer l's c_proj [n_embd, n_embd]
        self.adapters_c_attn = nn.ModuleList()
        self.adapters_c_proj = nn.ModuleList()
        self._hook_handles   = []

        for l in range(self._n_layers):
            attn_mod = base_model.transformer.h[l].attn

            # c_attn: Conv1D with weight [n_embd, 3*n_embd]
            adapter_ca = PolarAdapter(n_embd, 3 * n_embd, rank, landing_lambda)
            self.adapters_c_attn.append(adapter_ca)

            # c_proj: Conv1D with weight [n_embd, n_embd]
            adapter_cp = PolarAdapter(n_embd, n_embd, rank, landing_lambda)
            self.adapters_c_proj.append(adapter_cp)

            # Register output hooks on the Conv1D modules
            handle_ca = attn_mod.c_attn.register_forward_hook(
                _make_conv1d_hook(adapter_ca)
            )
            handle_cp = attn_mod.c_proj.register_forward_hook(
                _make_conv1d_hook(adapter_cp)
            )
            self._hook_handles.extend([handle_ca, handle_cp])

    def remove_hooks(self) -> None:
        """Remove all forward hooks (useful for cleanup)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        self.base.gradient_checkpointing_disable(**kwargs)

    @property
    def config(self):
        return self.base.config


def _make_conv1d_hook(adapter: PolarAdapter):
    """
    Factory for a Conv1D forward output hook.

    Conv1D forward: output = input @ weight + bias   (input: [B, T, m])
    Hook adds:      output + input @ ΔW              (ΔW: [m, n])
    Net result:     output = input @ (W + ΔW)

    The hook captures `adapter` by reference — always uses the current ΔW.
    """
    def hook(
        module,                    # Conv1D module (not used — ΔW from adapter)
        input:  Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        hidden = input[0]                  # [B, T, m]
        delta  = adapter.delta_weight()    # [m, n]
        return output + hidden @ delta

    return hook


# ──────────────────────────────────────────────────────────────────────────────
# PolarBaseline — BaseMethod implementation
# ──────────────────────────────────────────────────────────────────────────────

class PolarBaseline(BaseMethod):

    def _build_model(self, hf_name: str) -> PolarModel:
        """
        Load GPT-2 on CPU, create PolarModel with adapters and forward hooks.
        Base model weights start frozen (will be frozen in _configure_parameters).
        All adapter parameters start with requires_grad=False.
        Landing field hooks are registered in _configure_parameters, not here,
        because register_hook requires requires_grad=True.
        """
        lora_cfg       = self.cfg.get("method", {})
        rank           = lora_cfg.get("lora_rank", 8)
        landing_lambda = lora_cfg.get("landing_lambda", 0.1)

        base = GPT2LMHeadModel.from_pretrained(hf_name)
        base.eval()
        return PolarModel(base, rank=rank, landing_lambda=landing_lambda)

    def _configure_parameters(self) -> None:
        """
        1. Freeze the entire model (base weights + adapter params).
        2. Unfreeze X, Theta, Y for every adapter.
        3. Register landing field hooks on X and Y (requires requires_grad=True).
        """
        freeze_all(self.model)

        for l in range(self.model._n_layers):
            for adapter in (
                self.model.adapters_c_attn[l],
                self.model.adapters_c_proj[l],
            ):
                adapter.X.requires_grad_(True)
                adapter.Theta.requires_grad_(True)
                adapter.Y.requires_grad_(True)
                # Register landing field hooks AFTER setting requires_grad=True
                adapter.register_landing_hooks()

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Return W_0 + ΔW for the V and O projections, per head.

        c_attn ΔW: [n_embd, 3*n_embd] — V portion is columns [2*n_embd:]
        c_proj ΔW: [n_embd, n_embd]   — this is the O projection

        W_V_live = W_0_V + ΔW_V   then reshape to [H, n_embd, d_head]
        W_O_live = W_0_O + ΔW_O   then reshape to [H, d_head, n_embd]
        """
        dims = get_gpt2_dims(self.model.base)
        n, H, d = dims.n_embd, dims.n_heads, dims.d_head
        W_V_layers: List[torch.Tensor] = []
        W_O_layers: List[torch.Tensor] = []

        with torch.no_grad():
            for l in range(dims.n_layers):
                attn = self.model.base.transformer.h[l].attn

                # Base weights (frozen, in Conv1D layout [in, out])
                W0_attn = attn.c_attn.weight.detach()  # [n, 3n]
                W0_proj = attn.c_proj.weight.detach()  # [n, n]

                # Adapter deltas
                dW_attn = self.model.adapters_c_attn[l].delta_weight().detach()  # [n, 3n]
                dW_proj = self.model.adapters_c_proj[l].delta_weight().detach()  # [n, n]

                # V projection: columns [2n:3n] of c_attn
                W_V_live = (W0_attn + dW_attn)[:, 2*n:]  # [n, n]
                W_V = W_V_live.reshape(n, H, d).permute(1, 0, 2).contiguous()  # [H, n, d]

                # O projection: c_proj
                W_O_live = W0_proj + dW_proj              # [n, n]
                W_O = W_O_live.reshape(H, d, n).contiguous()                    # [H, d, n]

                W_V_layers.append(W_V)
                W_O_layers.append(W_O)

        return {"W_V": W_V_layers, "W_O": W_O_layers}

    def cleanup(self) -> None:
        """Remove forward hooks before cleanup to prevent dangling references."""
        if self.model is not None and hasattr(self.model, "remove_hooks"):
            self.model.remove_hooks()
        super().cleanup()

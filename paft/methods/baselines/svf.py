"""
SVF (M4) — Singular Value Fine-tuning.

Freeze the left and right singular vectors (U, Vh) of each head's W_V and W_O.
Train only the singular values sigma — a diagonal parameter in the SVD basis.

This is the most direct structural competitor to pure_paft:
    Both freeze rotational structure and tune only scalar magnitudes.
    Both train [n_layers × n_heads × 2 × d_head] = 18,432 parameters.

The comparison isolates one specific hypothesis:
    Is the polar eigenbasis a better inductive bias than the SVD singular basis
    for scaling-only adaptation of the OV circuit?

    If pure_paft > SVF: polar decomposition provides a better geometric basis.
    If SVF ≈ pure_paft: the specific factorisation doesn't matter much.
    If SVF > pure_paft: SVD basis is better (unexpected — would require explanation).

Trainable parameters:
    sigma_V  [H, d_head]  per layer — singular values of W_V
    sigma_O  [H, d_head]  per layer — singular values of W_O
    Total: 12 × 12 × 2 × 64 = 18,432  (GPT-2 small)

Non-negative constraint:
    Singular values are non-negative by definition. The optimiser is not
    constrained here — sigma can go negative during training. We allow this
    deliberately to match the unconstrained treatment of lambda in pure_paft,
    making the comparison fair. If sigma goes negative it is mathematically
    equivalent to a sign flip in U (which is frozen), so the effective
    weight W = U @ diag(sigma) @ Vh remains valid.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, freeze_all
from paft.model.svf_model import SVFModel


class SVFBaseline(BaseMethod):

    def _build_model(self, hf_name: str) -> SVFModel:
        """
        Load GPT-2 on CPU, run economy SVD for all heads, build SVFModel.
        All parameters start frozen (requires_grad=False).
        SVFModel._decompose_all() uses the two-pass pattern: extract all
        layers before replacing any, eliminating the post-surgery extraction bug.
        """
        base = GPT2LMHeadModel.from_pretrained(hf_name)
        base.eval()
        return SVFModel(base)

    def _configure_parameters(self) -> None:
        """
        Freeze everything, then unfreeze sigma_V and sigma_O in every layer.
        U_V, Vh_V, U_O, Vh_O, W_Q, W_K, biases all remain frozen.
        """
        freeze_all(self.model)
        for _, attn in self.model.iter_svf_attentions():
            attn.sigma_V.requires_grad_(True)
            attn.sigma_O.requires_grad_(True)

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Reconstruct W_V and W_O from current sigma and frozen U, Vh.
        Identical to the forward-pass reconstruction — no staleness risk.
        """
        W_V_layers: List[torch.Tensor] = []
        W_O_layers: List[torch.Tensor] = []
        with torch.no_grad():
            for _, attn in self.model.iter_svf_attentions():
                W_V_layers.append(attn.get_W_V_per_head())
                W_O_layers.append(attn.get_W_O_per_head())
        return {"W_V": W_V_layers, "W_O": W_O_layers}

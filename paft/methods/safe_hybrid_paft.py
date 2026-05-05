"""
SafeHybridPAFT (M11) — full S matrices plus all bias terms.

Trainable parameters:
    S_V    [n_heads, d_head, d_head]  per layer
    S_O    [n_heads, d_head, d_head]  per layer
    b_qkv  [3*n_embd]                 per layer
    b_o    [n_embd]                   per layer
    + all MLP biases and LayerNorm biases in the base model

This is the most expressive PAFT variant.  It combines:
    - Full geometric freedom (S is unconstrained, unlike pure where only lam moves)
    - Residual stream flexibility (biases allow per-position offsets)

Expected use:
    Primary PAFT result for large-shift domains (biomedical, code) where both
    geometric and residual adaptation are needed.  If safe_hybrid_paft does not
    outperform hybrid_paft, it suggests the residual stream is already stable
    under hybrid adaptation — a finding worth reporting.

Parameter count (GPT-2 small):
    1,179,648  (S matrices, same as hybrid_paft)
  + ~102,400   (biases)
  ≈ 1,282,048 total

The bias configuration is identical to safe_pure_paft — see that module for
the detailed justification of which biases are unfrozen and why.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, PAFTSnapshot, freeze_all
from paft.model.paft_model import PAFTModel
from paft.methods.safe_pure_paft import _is_non_attention_bias


class SafeHybridPAFT(BaseMethod):

    def _build_model(self, hf_name: str) -> PAFTModel:
        base = GPT2LMHeadModel.from_pretrained(hf_name)
        base.eval()
        model = PAFTModel(base)
        model.set_mode("hybrid")
        return model

    def _configure_parameters(self) -> None:
        """
        Freeze everything, then unfreeze S_V/S_O and all bias terms.
        Logic is the union of hybrid_paft and safe_pure_paft configurations.
        """
        freeze_all(self.model)

        for _, attn in self.model.iter_paft_attentions():
            # Geometric: full S matrices
            attn.S_V.requires_grad_(True)
            attn.S_O.requires_grad_(True)
            # Residual stream: attention biases
            attn.b_qkv.requires_grad_(True)
            attn.b_o.requires_grad_(True)

        # MLP, LayerNorm, and other non-attention biases in the base model
        for name, param in self.model.base.named_parameters():
            if _is_non_attention_bias(name):
                param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Live weights
    # ------------------------------------------------------------------

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        W_V_layers: List[torch.Tensor] = []
        W_O_layers: List[torch.Tensor] = []
        with torch.no_grad():
            for _, attn in self.model.iter_paft_attentions():
                W_V_layers.append(attn.get_W_V_per_head())
                W_O_layers.append(attn.get_W_O_per_head())
        return {"W_V": W_V_layers, "W_O": W_O_layers}

    # ------------------------------------------------------------------
    # PAFT snapshot  (same logic as hybrid_paft — eigendecompose current S)
    # ------------------------------------------------------------------

    def paft_snapshot(self) -> PAFTSnapshot:
        """
        Computes lam/EV by eigendecomposing the current S at snapshot time.
        Symmetrizes S before eigh to guard against asymmetric drift during training.
        """
        snap = PAFTSnapshot()

        with torch.no_grad():
            for _, attn in self.model.iter_paft_attentions():
                snap.Q_V.append(attn.Q_V.cpu())
                snap.Q_O.append(attn.Q_O.cpu())

                S_V = attn.S_V.detach().cpu()
                S_O = attn.S_O.detach().cpu()
                snap.S_V.append(S_V)
                snap.S_O.append(S_O)

                snap.EV_V.append(attn.EV_V.cpu())
                snap.EV_O.append(attn.EV_O.cpu())

                n_heads = S_V.shape[0]
                lam_V_list, lam_O_list = [], []
                for h in range(n_heads):
                    S_V_sym = (S_V[h] + S_V[h].T) / 2.0
                    S_O_sym = (S_O[h] + S_O[h].T) / 2.0
                    lv, _ = torch.linalg.eigh(S_V_sym)
                    lo, _ = torch.linalg.eigh(S_O_sym)
                    lam_V_list.append(lv.flip(0))
                    lam_O_list.append(lo.flip(0))

                snap.lam_V.append(torch.stack(lam_V_list))
                snap.lam_O.append(torch.stack(lam_O_list))

        return snap

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["paft_snapshot"] = self.paft_snapshot()
        return d
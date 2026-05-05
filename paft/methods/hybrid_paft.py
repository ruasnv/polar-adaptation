"""
HybridPAFT (M9) — train full S_V and S_O matrices.

Trainable parameters:
    S_V  [n_heads, d_head, d_head]  per layer — full V-side scaling matrix
    S_O  [n_heads, d_head, d_head]  per layer — full O-side scaling matrix

Forward reconstruction (mode='hybrid'):
    W_V_h = Q_V[h] @ S_V[h]        gradient flows directly through S_V
    W_O_h = S_O[h] @ Q_O[h]        gradient flows directly through S_O

S_V and S_O are unconstrained nn.Parameters — the optimizer can move them
anywhere.  They are initialised to the pretrained polar S, so training starts
from the correct geometric point.  Q_V, Q_O remain frozen throughout.

Parameter count (GPT-2 small, 12L 12H d_head=64):
    12 layers × 12 heads × 2 sides × 64 × 64 = 1,179,648

Note on symmetry:
    S_V and S_O are initialised symmetric PSD (from polar decomposition) but
    the unconstrained parameterisation does not enforce this during training.
    Symmetry is monitored via the geometric audit but not enforced.
    If enforced symmetry is desired, use (S + S^T)/2 in the forward pass
    inside PAFTAttention._get_S_V() — not done here to avoid constraining
    the gradient landscape and to make the comparison with pure_paft cleaner.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, PAFTSnapshot, freeze_all
from paft.model.paft_model import PAFTModel


class HybridPAFT(BaseMethod):

    def _build_model(self, hf_name: str) -> PAFTModel:
        base = GPT2LMHeadModel.from_pretrained(hf_name)
        base.eval()
        model = PAFTModel(base)
        model.set_mode("hybrid")
        return model

    def _configure_parameters(self) -> None:
        """
        Freeze everything, then unfreeze S_V and S_O in every layer.
        lam_V and lam_O remain frozen (not used in hybrid forward, but
        stored for snapshot — keeping them frozen prevents gradient accumulation
        on unused parameters which would waste memory).
        """
        freeze_all(self.model)
        for _, attn in self.model.iter_paft_attentions():
            attn.S_V.requires_grad_(True)
            attn.S_O.requires_grad_(True)

    # ------------------------------------------------------------------
    # Live weights for geometric health
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
    # PAFT snapshot
    # ------------------------------------------------------------------

    def paft_snapshot(self) -> PAFTSnapshot:
        """
        For hybrid, lam and EV are not directly trained but are computed
        at snapshot time via eigendecomposition of the current S.  This lets
        analysis scripts compare eigenvalue shifts across pure and hybrid
        without special-casing the method.
        """
        snap = PAFTSnapshot()

        with torch.no_grad():
            for _, attn in self.model.iter_paft_attentions():
                snap.Q_V.append(attn.Q_V.cpu())
                snap.Q_O.append(attn.Q_O.cpu())

                S_V = attn.S_V.detach().cpu()   # [n_heads, d_head, d_head]
                S_O = attn.S_O.detach().cpu()
                snap.S_V.append(S_V)
                snap.S_O.append(S_O)

                snap.EV_V.append(attn.EV_V.cpu())  # frozen initial eigenvectors
                snap.EV_O.append(attn.EV_O.cpu())

                # Compute current eigenvalues from current S via eigh.
                # eigh is safe for symmetric matrices; S may have drifted from
                # symmetry so we symmetrize first for numerical stability.
                n_heads, d, _ = S_V.shape
                lam_V_list, lam_O_list = [], []
                for h in range(n_heads):
                    S_V_sym = (S_V[h] + S_V[h].T) / 2.0
                    S_O_sym = (S_O[h] + S_O[h].T) / 2.0
                    lv, _ = torch.linalg.eigh(S_V_sym)
                    lo, _ = torch.linalg.eigh(S_O_sym)
                    lam_V_list.append(lv.flip(0))   # descending
                    lam_O_list.append(lo.flip(0))

                snap.lam_V.append(torch.stack(lam_V_list))  # [n_heads, d_head]
                snap.lam_O.append(torch.stack(lam_O_list))

        return snap

    # ------------------------------------------------------------------
    # Checkpoint extras
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["paft_snapshot"] = self.paft_snapshot()
        return d
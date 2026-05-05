"""
PurePAFT (M8) — train only the eigenvalues of S_V and S_O.

Trainable parameters:
    lam_V  [n_heads, d_head]  per layer — eigenvalues of V-side scaling matrix
    lam_O  [n_heads, d_head]  per layer — eigenvalues of O-side scaling matrix

Forward reconstruction (mode='pure'):
    S_V_h = EV_V[h] @ diag(lam_V[h]) @ EV_V[h]^T
    W_V_h = Q_V[h]  @ S_V_h
    S_O_h = EV_O[h] @ diag(lam_O[h]) @ EV_O[h]^T
    W_O_h = S_O_h   @ Q_O[h]

Q_V, Q_O, EV_V, EV_O are frozen buffers — only lam_V/lam_O have gradients.
Gradient flows: loss → lam → S (via diag recon) → W → attention → loss.

Parameter count (GPT-2 small, 12L 12H d_head=64):
    12 layers × 12 heads × 2 sides × 64 = 18,432
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, PAFTSnapshot, freeze_all
from paft.model.paft_model import PAFTModel


class PurePAFT(BaseMethod):

    def _build_model(self, hf_name: str) -> PAFTModel:
        """
        Load pretrained GPT-2 on CPU, polar-decompose every (layer, head),
        wrap in PAFTModel.  All parameters start frozen (requires_grad=False).
        Full decomposition happens here on CPU before any .to(device) call.
        """
        base = GPT2LMHeadModel.from_pretrained(hf_name)
        base.eval()  # not strictly needed but prevents accidental dropout state
        model = PAFTModel(base)
        model.set_mode("pure")
        return model

    def _configure_parameters(self) -> None:
        """
        Freeze everything, then unfreeze only lam_V and lam_O in every
        PAFTAttention layer.  All other parameters — Q, S, EV, W_Q, W_K,
        biases, embeddings, LM head — remain frozen.
        """
        freeze_all(self.model)
        for _, attn in self.model.iter_paft_attentions():
            attn.lam_V.requires_grad_(True)
            attn.lam_O.requires_grad_(True)

    # ------------------------------------------------------------------
    # Live weights for geometric health
    # ------------------------------------------------------------------

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Reconstruct W_V and W_O from current lam_V/lam_O and frozen Q/EV.
        Reconstruction is identical to the forward pass — no staleness risk.
        """
        W_V_layers: List[torch.Tensor] = []
        W_O_layers: List[torch.Tensor] = []

        with torch.no_grad():
            for _, attn in self.model.iter_paft_attentions():
                # [n_heads, n_embd, d_head] and [n_heads, d_head, n_embd]
                W_V_layers.append(attn.get_W_V_per_head())
                W_O_layers.append(attn.get_W_O_per_head())

        return {"W_V": W_V_layers, "W_O": W_O_layers}

    # ------------------------------------------------------------------
    # PAFT snapshot  (saved every epoch by checkpointing schema)
    # ------------------------------------------------------------------

    def paft_snapshot(self) -> PAFTSnapshot:
        """
        Collect Q, S, EV, lam for every (layer, head).
        S is reconstructed from current lam so the snapshot is consistent
        regardless of whether mode is pure or hybrid.
        All tensors moved to CPU before returning.
        """
        snap = PAFTSnapshot()

        with torch.no_grad():
            for _, attn in self.model.iter_paft_attentions():
                snap.Q_V.append(attn.Q_V.cpu())
                snap.Q_O.append(attn.Q_O.cpu())

                # Reconstruct current S from lam (mode=pure)
                S_V = attn._get_S_V().cpu()   # [n_heads, d_head, d_head]
                S_O = attn._get_S_O().cpu()
                snap.S_V.append(S_V)
                snap.S_O.append(S_O)

                snap.EV_V.append(attn.EV_V.cpu())
                snap.EV_O.append(attn.EV_O.cpu())
                snap.lam_V.append(attn.lam_V.detach().cpu())
                snap.lam_O.append(attn.lam_O.detach().cpu())

        return snap

    # ------------------------------------------------------------------
    # Checkpoint extras
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["paft_snapshot"] = self.paft_snapshot()
        return d
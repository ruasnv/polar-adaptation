"""
SafePurePAFT (M10) — eigenvalues of S plus all bias terms.

Trainable parameters:
    lam_V   [n_heads, d_head]  per layer   (same as pure_paft)
    lam_O   [n_heads, d_head]  per layer
    b_qkv   [3*n_embd]         per layer   attention QKV bias
    b_o     [n_embd]           per layer   attention output projection bias
    + all MLP biases and LayerNorm biases in the base model

The bias terms give the residual stream a direct additive degree of freedom.
Without them, every token at every position sees the same scaling applied to
its V and O projections — the model must encode all adaptation in the geometric
change to S.  Biases allow per-position offsets in the residual stream, which
is essential when the target domain has a very different token distribution
from the pretraining distribution (large-shift domains: biomedical, code).

This is the primary comparison against BitFit:
    safe_pure_paft = geometric scaling (via lam) + residual offsets (via bias)
    bitfit         = residual offsets only (via bias)
    pure_paft      = geometric scaling only (no bias)

The difference in performance between safe_pure_paft and pure_paft isolates
how much the bias terms contribute.  The difference between safe_pure_paft
and bitfit isolates how much the geometric component contributes.

Parameter count (GPT-2 small):
    18,432  (lam, same as pure_paft)
  + ~102,400 (biases: attn + mlp + ln across 12 layers)
  ≈ 120,832 total
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, PAFTSnapshot, freeze_all
from paft.model.paft_model import PAFTModel, PAFTAttention


class SafePurePAFT(BaseMethod):

    def _build_model(self, hf_name: str) -> PAFTModel:
        base = GPT2LMHeadModel.from_pretrained(hf_name)
        base.eval()
        model = PAFTModel(base)
        model.set_mode("pure")
        return model

    def _configure_parameters(self) -> None:
        """
        Freeze everything.  Then unfreeze:
          - lam_V, lam_O in every PAFTAttention layer
          - b_qkv, b_o in every PAFTAttention layer
          - every bias in the base GPT-2 model (MLP, LayerNorm, etc.)
        """
        freeze_all(self.model)

        for _, attn in self.model.iter_paft_attentions():
            # Geometric eigenvalue terms
            attn.lam_V.requires_grad_(True)
            attn.lam_O.requires_grad_(True)
            # Attention biases (b_qkv covers Q, K, V bias; b_o covers output proj)
            attn.b_qkv.requires_grad_(True)
            attn.b_o.requires_grad_(True)

        # MLP biases, LayerNorm biases, embedding biases live in the base model.
        # We iterate named_parameters to catch everything with "bias" in the name
        # that isn't already a PAFTAttention parameter (those are handled above).
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
    # PAFT snapshot  (identical logic to pure_paft)
    # ------------------------------------------------------------------

    def paft_snapshot(self) -> PAFTSnapshot:
        snap = PAFTSnapshot()
        with torch.no_grad():
            for _, attn in self.model.iter_paft_attentions():
                snap.Q_V.append(attn.Q_V.cpu())
                snap.Q_O.append(attn.Q_O.cpu())
                snap.S_V.append(attn._get_S_V().cpu())
                snap.S_O.append(attn._get_S_O().cpu())
                snap.EV_V.append(attn.EV_V.cpu())
                snap.EV_O.append(attn.EV_O.cpu())
                snap.lam_V.append(attn.lam_V.detach().cpu())
                snap.lam_O.append(attn.lam_O.detach().cpu())
        return snap

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["paft_snapshot"] = self.paft_snapshot()
        return d


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_non_attention_bias(name: str) -> bool:
    """
    Return True for bias parameters that live in the base GPT-2 model and are
    NOT attention projection biases (which are stored separately in PAFTAttention
    as b_qkv and b_o and handled by the loop over iter_paft_attentions()).

    We unfreeze everything with "bias" in the name except for the c_attn and
    c_proj parameters — those are never updated because PAFTAttention's b_qkv
    and b_o have replaced them structurally.

    Concretely, what gets unfrozen here:
        transformer.h[i].mlp.c_fc.bias       [4*n_embd]
        transformer.h[i].mlp.c_proj.bias     [n_embd]
        transformer.h[i].ln_1.bias           [n_embd]
        transformer.h[i].ln_2.bias           [n_embd]
        transformer.ln_f.bias                [n_embd]
        # Note: GPT-2 has no embedding bias; lm_head bias is optional
    """
    if "bias" not in name:
        return False
    # c_attn and c_proj biases are managed by PAFTAttention — skip them
    if "attn.c_attn" in name or "attn.c_proj" in name:
        return False
    return True
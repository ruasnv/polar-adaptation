"""
BitFit (M3) — bias terms only.

Reference: Ben Zaken et al. 2021, "BitFit: Simple Parameter-efficient
Fine-tuning for Transformer-based Masked Language-models."

All weight matrices are frozen.  Only bias vectors are trained.
Biases provide per-position additive offsets in the residual stream without
touching the geometric structure of the weight matrices.

This is the primary comparison for the safe_* PAFT variants:
    safe_pure_paft  = λ (geometry) + biases  →  does geometry help beyond biases?
    safe_hybrid_paft = S (geometry) + biases  →  same question
    bitfit           = biases only             →  pure residual stream baseline

Biases trained in GPT-2:
    c_attn.bias   [3*n_embd]  per layer  — Q, K, V projection biases
    c_proj.bias   [n_embd]    per layer  — O projection bias
    mlp.c_fc.bias [4*n_embd]  per layer
    mlp.c_proj.bias [n_embd]  per layer
    ln_1.bias     [n_embd]    per layer  — first LayerNorm
    ln_2.bias     [n_embd]    per layer  — second LayerNorm
    ln_f.bias     [n_embd]              — final LayerNorm

Parameter count (GPT-2 small, 12 layers, n_embd=768):
    Per layer: 3*768 + 768 + 3072 + 768 + 768 + 768 = 7,680
    Final LN:  768
    Total:     12 * 7,680 + 768 = 92,928 ≈ 93K

Note: GPT-2 has no lm_head bias and no embedding bias.

Geometric health:
    W_V and W_O are never modified — their geometric health metrics are
    identical to Frozen.  The comparison isolates how much the bias
    terms contribute to task performance.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, freeze_all
from paft.model.extractor import get_gpt2_dims, extract_wv_all_heads, extract_wo_all_heads


class BitFit(BaseMethod):

    def _build_model(self, hf_name: str) -> GPT2LMHeadModel:
        return GPT2LMHeadModel.from_pretrained(hf_name)

    def _configure_parameters(self) -> None:
        """Freeze all weights; unfreeze every parameter whose name contains 'bias'."""
        freeze_all(self.model)
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.requires_grad_(True)

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        W_V and W_O are frozen — identical to pretrained every call.
        Returned for geometric health comparison with other methods.
        """
        dims = get_gpt2_dims(self.model)
        W_V, W_O = [], []
        with torch.no_grad():
            for l in range(dims.n_layers):
                W_V.append(extract_wv_all_heads(self.model, l, dims))
                W_O.append(extract_wo_all_heads(self.model, l, dims))
        return {"W_V": W_V, "W_O": W_O}

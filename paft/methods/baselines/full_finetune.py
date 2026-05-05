"""
FullFinetune (M2) — all ~117M / ~345M parameters trainable.

Upper-bound baseline.  Maximum capacity, maximum parameter count.
Sets the performance ceiling — no PEFT method should be expected to
exceed this, but methods that come close with far fewer parameters
make the paper's case.

Gradient checkpointing is critical for GPT-2 medium on 6 GB VRAM.
BaseMethod.build() calls gradient_checkpointing_enable() automatically
before .to(device).  GPT2LMHeadModel supports this natively.

Trainable parameters:
    GPT-2 small:  ~117M
    GPT-2 medium: ~345M

Geometric health notes:
    Full fine-tuning is the only method that can freely modify Q, K, MLP
    and all other weights — not just the OV circuit.  Its geometric health
    metrics are therefore not directly comparable to other methods (which
    freeze most weights), but are still useful as a reference for how much
    the weights move under unconstrained optimisation.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, freeze_all, unfreeze
from paft.model.extractor import get_gpt2_dims, extract_wv_all_heads, extract_wo_all_heads


class FullFinetune(BaseMethod):

    def _build_model(self, hf_name: str) -> GPT2LMHeadModel:
        return GPT2LMHeadModel.from_pretrained(hf_name)

    def _configure_parameters(self) -> None:
        # Freeze first (mandatory BaseMethod pattern), then unfreeze everything.
        # This makes the pattern explicit and lets the assertion in build()
        # correctly count trainable params.
        freeze_all(self.model)
        unfreeze(self.model.parameters())

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Read current W_V and W_O directly from the model weights.
        These change every step — we always read the current state.
        """
        dims = get_gpt2_dims(self.model)
        W_V, W_O = [], []
        with torch.no_grad():
            for l in range(dims.n_layers):
                W_V.append(extract_wv_all_heads(self.model, l, dims))
                W_O.append(extract_wo_all_heads(self.model, l, dims))
        return {"W_V": W_V, "W_O": W_O}

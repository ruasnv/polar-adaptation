"""
Frozen (M1) — all parameters frozen, zero adaptation.

Lower-bound baseline.  Measures pretrained GPT-2 performance on each domain
task with no fine-tuning.  Any method that cannot beat this is broken.

Geometric health metrics are still computed every epoch — not because the
weights change (they don't), but to establish the pretrained model's baseline
stable rank, entropy, and condition number for every layer.  These values
are the reference against which all other methods' geometric degradation
or preservation is measured.

Trainable parameters: 0
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel

from paft.methods.base import BaseMethod, freeze_all
from paft.model.extractor import get_gpt2_dims, extract_wv_all_heads, extract_wo_all_heads


class Frozen(BaseMethod):

    def _build_model(self, hf_name: str) -> GPT2LMHeadModel:
        return GPT2LMHeadModel.from_pretrained(hf_name)

    def _configure_parameters(self) -> None:
        freeze_all(self.model)

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Return the pretrained W_V and W_O — identical on every call since
        nothing is being trained.  Used to establish the geometric baseline.
        """
        dims = get_gpt2_dims(self.model)
        W_V, W_O = [], []
        with torch.no_grad():
            for l in range(dims.n_layers):
                W_V.append(extract_wv_all_heads(self.model, l, dims))
                W_O.append(extract_wo_all_heads(self.model, l, dims))
        return {"W_V": W_V, "W_O": W_O}

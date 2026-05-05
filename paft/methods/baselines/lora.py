"""
LoRA baselines (M5, M6) — low-rank additive updates.

Reference: Hu et al. 2022, "LoRA: Low-Rank Adaptation of Large Language Models."

Parameterisation: ΔW = B @ A  where B [d_out, r], A [r, d_in], rank r ≪ d_in.
The base weight W_0 is frozen; only A and B are trained.
The adapted weight is W_0 + (alpha/r) * B @ A.

Two variants, differing only in rank:
    lora_r8:   rank=8,  alpha=16   — matched parameter count to polar_r8
    lora_r64:  rank=64, alpha=128  — higher expressiveness bracket

Target modules: c_attn and c_proj (the QKV fused projection and output projection).
MLP and embedding weights are not adapted — same scope as all PAFT variants.

Parameter count (GPT-2 small, n_embd=768, n_layers=12):
    c_attn [768, 2304]: A [r, 768], B [2304, r]  →  r * (768 + 2304) = 3072*r
    c_proj [768,  768]: A [r, 768], B [ 768, r]  →  r * (768 +  768) = 1536*r
    Per layer: (3072 + 1536) * r = 4608 * r
    Total: 12 * 4608 * r
        r=8:   12 * 4608 * 8  = 442,368  ≈ 442K
        r=64:  12 * 4608 * 64 = 3,538,944 ≈ 3.5M

Note: PEFT handles GPT-2 Conv1D (weight [in, out]) transparently via the
fan_in_fan_out flag.  The effective weight update is correctly applied in
Conv1D convention.

Geometric health — get_live_WV_WO:
    Returns W_0 + ΔW by temporarily merging LoRA weights into the base model,
    reading the merged weights, then unmerging.  merge/unmerge is O(n_params)
    but only called once per epoch during eval — acceptable overhead.
    merge() handles Conv1D convention internally.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

from paft.methods.base import BaseMethod, freeze_all
from paft.model.extractor import get_gpt2_dims

logger = logging.getLogger(__name__)


def _require_peft() -> None:
    if not _PEFT_AVAILABLE:
        raise ImportError(
            "peft is required for LoRA baselines. "
            "Install with: pip install peft>=0.9.0"
        )


class LoRABaseline(BaseMethod):
    """
    Single LoRA class — rank and alpha are read from self.cfg.

    Expected cfg keys (under cfg['method']['lora']):
        rank:    int   (e.g. 8 or 64)
        alpha:   int   (e.g. 16 or 128)
        dropout: float (default 0.05)

    Instantiated as either lora_r8 or lora_r64 by the method factory,
    which sets the appropriate cfg before calling build().
    """

    def _build_model(self, hf_name: str):
        _require_peft()

        lora_cfg = self.cfg.get("method", {}).get("lora", {})
        rank     = lora_cfg.get("rank", 8)
        alpha    = lora_cfg.get("alpha", rank * 2)
        dropout  = lora_cfg.get("dropout", 0.05)

        logger.info(
            f"[{self.method_name}] LoRA rank={rank} alpha={alpha} dropout={dropout}"
        )

        base = GPT2LMHeadModel.from_pretrained(hf_name)

        peft_config = LoraConfig(
            task_type   = TaskType.CAUSAL_LM,
            r           = rank,
            lora_alpha  = alpha,
            lora_dropout= dropout,
            # c_attn: fused QKV [n_embd, 3*n_embd] — targets V and QK
            # c_proj: output projection [n_embd, n_embd]
            # PEFT handles GPT-2's Conv1D convention via fan_in_fan_out detection.
            target_modules = ["c_attn", "c_proj"],
            bias           = "none",   # biases not adapted — matches PAFT scope
        )

        return get_peft_model(base, peft_config)

    def _configure_parameters(self) -> None:
        """
        PEFT's get_peft_model already set requires_grad correctly.
        We validate rather than override — calling freeze_all then re-unfreezing
        LoRA params would require knowing PEFT's internal parameter names, which
        is fragile across PEFT versions.

        Invariant: at least one LoRA parameter must be trainable.
        """
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                f"[{self.method_name}] PEFT wrapping produced no trainable parameters. "
                "Check that target_modules matches GPT-2's layer names."
            )
        logger.debug(
            f"[{self.method_name}] {len(trainable)} trainable parameter tensors "
            f"(PEFT-configured)"
        )

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Return W_0 + ΔW for every (layer, head) by temporarily merging
        the LoRA adapter weights into the base model weights.

        merge_adapter() modifies base weights in-place:
            W_merged = W_0 + (alpha/r) * B @ A    (PEFT handles Conv1D correctly)
        unmerge_adapter() restores W_0 in-place.

        Called once per epoch in no_grad context — merge/unmerge overhead
        is acceptable (~5 ms for GPT-2 small).
        """
        # Access the wrapped HuggingFace model
        # PeftModel structure: self.model.base_model.model is GPT2LMHeadModel
        base_gpt2 = self.model.base_model.model

        W_V_layers: List[torch.Tensor] = []
        W_O_layers: List[torch.Tensor] = []

        with torch.no_grad():
            # Merge LoRA deltas into base weights
            self.model.merge_adapter()

            dims = get_gpt2_dims(base_gpt2)
            n    = dims.n_embd
            H    = dims.n_heads
            d    = dims.d_head

            for l in range(dims.n_layers):
                attn = base_gpt2.transformer.h[l].attn

                # After merge, base_layer.weight holds W_0 + ΔW in Conv1D layout
                # c_attn: [n_embd, 3*n_embd]
                w_attn = _get_base_weight(attn.c_attn)   # [n, 3n]
                w_proj = _get_base_weight(attn.c_proj)   # [n, n]

                # V block: columns [2n:3n] → [H, n, d]
                W_V = (
                    w_attn[:, 2*n:]
                    .clone()
                    .reshape(n, H, d)
                    .permute(1, 0, 2)
                    .contiguous()
                )
                # O block: [n, n] → [H, d, n]
                W_O = w_proj.clone().reshape(H, d, n).contiguous()

                W_V_layers.append(W_V)
                W_O_layers.append(W_O)

            # Restore trainable LoRA parameters
            self.model.unmerge_adapter()

        return {"W_V": W_V_layers, "W_O": W_O_layers}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_base_weight(layer) -> torch.Tensor:
    """
    Get the weight tensor from a PEFT-wrapped or plain Conv1D layer.

    After merge_adapter(), PEFT layers expose the merged weight via
    .base_layer.weight (for LoraLinear) or directly as .weight (plain Conv1D).
    """
    if hasattr(layer, "base_layer"):
        return layer.base_layer.weight.detach()
    return layer.weight.detach()

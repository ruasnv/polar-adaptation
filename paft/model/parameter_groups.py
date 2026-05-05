"""
parameter_groups.py — shim that re-exports configure_* functions.

Historical note: early versions of PAFT had parameter configuration logic
here.  It has since moved into each method's _configure_parameters() method
(see paft/methods/{pure_paft,hybrid_paft,safe_pure_paft,safe_hybrid_paft}.py).

This file now exists only so that any older method files that still import
    from paft.model.parameter_groups import configure_hybrid_paft
continue to work without modification.  The implementations here delegate
to the correct new PAFTModel API.

New code should NOT import from this module — call _configure_parameters()
through the method class instead.
"""

from __future__ import annotations

import logging
from typing import List

import torch.nn as nn

from paft.model.paft_model import PAFTModel, PAFTAttention

# Alias so any stale import of PAFTAttentionLayer still resolves
PAFTAttentionLayer = PAFTAttention

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper — freeze all, then unfreeze the given attr names on every
# PAFTAttention layer and optionally non-attention biases in base model.
# ──────────────────────────────────────────────────────────────────────────────

def _configure(
    model:           PAFTModel,
    mode:            str,
    attn_param_names: List[str],
    include_base_biases: bool = False,
) -> List[nn.Parameter]:
    """
    Generic configuration helper used by all four configure_* functions.

    Args:
        model:              PAFTModel instance (already built, on device).
        mode:               'pure' or 'hybrid' — sets PAFTAttention.mode.
        attn_param_names:   Attribute names on PAFTAttention to unfreeze
                            (e.g. ['lam_V', 'lam_O']).
        include_base_biases: If True, also unfreeze non-attention bias params
                            in the base GPT-2 model (MLP, LayerNorm biases).
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. Set mode on all attention layers
    model.set_mode(mode)

    # 3. Unfreeze the requested PAFTAttention parameters
    for _, attn in model.iter_paft_attentions():
        for attr in attn_param_names:
            param = getattr(attn, attr, None)
            if param is None:
                raise AttributeError(
                    f"PAFTAttention has no attribute '{attr}'. "
                    f"Available params: lam_V, lam_O, S_V, S_O, b_qkv, b_o"
                )
            param.requires_grad_(True)

    # 4. Optionally unfreeze non-attention biases in the base model
    if include_base_biases:
        for name, param in model.base.named_parameters():
            if "bias" in name and "attn.c_attn" not in name and "attn.c_proj" not in name:
                param.requires_grad_(True)

    return [p for p in model.parameters() if p.requires_grad]


# ──────────────────────────────────────────────────────────────────────────────
# Public API — kept for backward compatibility with older method files
# ──────────────────────────────────────────────────────────────────────────────

def configure_pure_paft(model: PAFTModel) -> List[nn.Parameter]:
    """Train only lam_V and lam_O (eigenvalues of S). Mode: pure."""
    return _configure(model, "pure", ["lam_V", "lam_O"])


def configure_hybrid_paft(model: PAFTModel) -> List[nn.Parameter]:
    """Train full S_V and S_O matrices. Mode: hybrid."""
    return _configure(model, "hybrid", ["S_V", "S_O"])


def configure_safe_pure_paft(model: PAFTModel) -> List[nn.Parameter]:
    """Train lam_V, lam_O + all bias terms. Mode: pure."""
    return _configure(model, "pure", ["lam_V", "lam_O", "b_qkv", "b_o"],
                      include_base_biases=True)


def configure_safe_hybrid_paft(model: PAFTModel) -> List[nn.Parameter]:
    """Train S_V, S_O + all bias terms. Mode: hybrid."""
    return _configure(model, "hybrid", ["S_V", "S_O", "b_qkv", "b_o"],
                      include_base_biases=True)


def configure_parameters(model: PAFTModel, method_name: str) -> List[nn.Parameter]:
    """Dispatch by method name string."""
    dispatch = {
        "pure_paft":        configure_pure_paft,
        "hybrid_paft":      configure_hybrid_paft,
        "safe_pure_paft":   configure_safe_pure_paft,
        "safe_hybrid_paft": configure_safe_hybrid_paft,
    }
    if method_name not in dispatch:
        raise ValueError(
            f"Unknown PAFT method '{method_name}'. Available: {list(dispatch)}"
        )
    return dispatch[method_name](model)
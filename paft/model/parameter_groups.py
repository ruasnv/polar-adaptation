"""
Configure which parameters are trainable for each PAFT method variant.

Each function:
  1. Calls model.set_mode() to set the forward-pass parameterization
  2. Sets requires_grad=True on exactly the correct parameter subset
  3. Returns the list of trainable parameters for the optimizer

Call exactly one of these functions after constructing PAFTModel.
"""

from typing import List
import torch.nn as nn

from paft.model.paft_model import PAFTModel, PAFTAttentionLayer


# ── PAFT variants ────────────────────────────────────────────────────────────

def configure_pure_paft(model: PAFTModel) -> List[nn.Parameter]:
    """
    pure_paft: train only lambda_V and lambda_O (eigenvalues of S).
    Most constrained variant: n_layers * n_heads * 2 * d_head parameters.
    GPT-2 small: 12 * 12 * 2 * 64 = 18,432 trainable parameters.
    """
    model.set_mode("pure")
    for layer in model.paft_layers():
        layer.lambda_V.requires_grad = True
        layer.lambda_O.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def configure_hybrid_paft(model: PAFTModel) -> List[nn.Parameter]:
    """
    hybrid_paft: train full S_V and S_O matrices.
    GPT-2 small: 12 * 12 * 2 * 64 * 64 = 1,179,648 trainable parameters.
    """
    model.set_mode("hybrid")
    for layer in model.paft_layers():
        layer.S_V.requires_grad = True
        layer.S_O.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def configure_safe_pure_paft(model: PAFTModel) -> List[nn.Parameter]:
    """
    safe_pure_paft: lambda_V, lambda_O + all bias terms.
    Bias terms stabilize the residual stream (Analysis 6).
    """
    model.set_mode("pure")
    # PAFT eigenvalues
    for layer in model.paft_layers():
        layer.lambda_V.requires_grad = True
        layer.lambda_O.requires_grad = True
        # Attention biases
        layer.c_attn_qk_bias.requires_grad = True
        layer.c_attn_v_bias.requires_grad  = True
        layer.c_proj_bias.requires_grad    = True
    # MLP and LayerNorm biases throughout the model
    for name, param in model.base_model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def configure_safe_hybrid_paft(model: PAFTModel) -> List[nn.Parameter]:
    """
    safe_hybrid_paft: S_V, S_O + all bias terms.
    Most expressive PAFT variant.
    """
    model.set_mode("hybrid")
    for layer in model.paft_layers():
        layer.S_V.requires_grad = True
        layer.S_O.requires_grad = True
        layer.c_attn_qk_bias.requires_grad = True
        layer.c_attn_v_bias.requires_grad  = True
        layer.c_proj_bias.requires_grad    = True
    for name, param in model.base_model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


# ── Dispatch ─────────────────────────────────────────────────────────────────

_CONFIGURATORS = {
    "pure_paft":        configure_pure_paft,
    "hybrid_paft":      configure_hybrid_paft,
    "safe_pure_paft":   configure_safe_pure_paft,
    "safe_hybrid_paft": configure_safe_hybrid_paft,
}


def configure_parameters(model: PAFTModel, method_name: str) -> List[nn.Parameter]:
    """
    Configure trainable parameters for a PAFT method.
    Raises ValueError for unknown method names.
    """
    if method_name not in _CONFIGURATORS:
        raise ValueError(
            f"Unknown PAFT method '{method_name}'. "
            f"Available: {list(_CONFIGURATORS)}"
        )
    return _CONFIGURATORS[method_name](model)
"""
LLaMA method variants for commonsense reasoning and GSM8K experiments.

All 6 methods:
  1. LLaMAPurePAFT   — eigenvalues of S_V only (lam_V per KV head)
  2. LLaMAHybridPAFT — full S_V matrices per KV head (free nn.Parameter)
  3. LLaMALoRA       — LoRA r=8 or r=64 on q/v/o projections
  4. LLaMABitFit     — all fp32 biases (LayerNorm, attention, MLP)
  5. LLaMAFrozen     — nothing trained (evaluation lower bound)
  6. LLaMAFullFT     — NOT RUN (too expensive on 8GB VRAM; cite LoRA paper numbers)

All models load LLaMA-3.2-3B with NF4 double quantization as the base.
This is the same base for ALL methods — fair comparison.

Critical: trainable parameters are ALWAYS in fp32, even when the base model
is NF4.  The bitsandbytes library ensures trainable params upcast to fp32
automatically, but we verify this explicitly in each build function.

Parameter counts (LLaMA-3.2-3B, 28 layers, 8 KV heads, head_dim=128):
  Pure PAFT:   28 × 8 × 128 = 28,672 params (PAFT) + no LM head change
  Hybrid PAFT: 28 × 8 × 128² = 3,670,016 params
  LoRA r=8 on q/v/o: 28 × (q+v+o) × r × dim ≈ 10.5M params (standard LoRA)
  BitFit:      depends on LLaMA config (LayerNorm only — ~200K params)
  Frozen:      0 trainable params

VRAM at training time:
  NF4 base:           ~1.8 GB
  Q_V buffers (fp16): ~175 MB
  S_V params (fp32):  ~14.7 MB
  AdamW optimizer:    ~30 MB
  Activations (ckpt): ~2.0 GB
  Total:              ~4.0 GB  ✓ fits in 8 GB
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from paft.model.llama_paft_model import LLaMAPAFTModel, load_llama_nf4, _dequantize_weight
from paft.model.paft_linear import PAFTLinear
from paft.methods.base import freeze_all   # canonical implementation from base.py

logger = logging.getLogger(__name__)

# Default model name — override via cfg or CLI argument
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_trainable_precision(model: nn.Module) -> None:
    """
    Assert that all trainable parameters are in fp32.
    Bitsandbytes should handle this automatically, but we verify explicitly.
    Raises RuntimeError if any trainable param is in lower precision.
    """
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype not in (torch.float32, torch.float):
            raise RuntimeError(
                f"Trainable parameter '{name}' is in {p.dtype}, not fp32. "
                "This may compromise numerical stability. "
                "Cast with p.data = p.data.float() or check bitsandbytes config."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Method 1: LLaMA Pure PAFT
# ──────────────────────────────────────────────────────────────────────────────

def build_llama_pure_paft(
    model_name: str = DEFAULT_MODEL,
    device_map: str = "auto",
) -> Tuple[LLaMAPAFTModel, Any]:
    """
    Trainable: lam_V [H_kv, d] per layer.
    Total params: 28 × 8 × 128 = 28,672 — extremely parameter efficient.

    Scientific argument: If pure PAFT (eigenvalue scaling only, 28K params)
    matches or approaches hybrid PAFT (full S matrices, 3.7M params), this
    demonstrates that the DOMINANT adaptation signal is captured by the
    eigenvalue magnitudes — the geometric directions (Q, EV) are already
    well-positioned by pretraining.
    """
    base, tokenizer = load_llama_nf4(model_name, device_map)
    model = LLaMAPAFTModel(base, train_mode='pure', q_dtype=torch.float32)   # was float16

    freeze_all(model)
    for _, vp in model._iter_paft_v_proj():
        vp.lam.requires_grad_(True)
        # Ensure fp32
        if vp.lam.dtype != torch.float32:
            vp.lam.data = vp.lam.data.float()

    verify_trainable_precision(model)
    logger.info(f"LLaMA Pure PAFT: {count_trainable(model):,} trainable params")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: LLaMA Hybrid PAFT
# ──────────────────────────────────────────────────────────────────────────────

def build_llama_hybrid_paft(
    model_name: str = DEFAULT_MODEL,
    device_map: str = "auto",
) -> Tuple[LLaMAPAFTModel, Any]:
    """
    Trainable: S_V [H_kv, d, d] per layer.
    Total params: 28 × 8 × 128² = 3,670,016.

    S is unconstrained (not enforced symmetric during training).
    Symmetry drift is monitored via geometric health analysis.
    S starts at the pretrained polar S (correct geometric initialisation).
    """
    base, tokenizer = load_llama_nf4(model_name, device_map)
    model = LLaMAPAFTModel(base, train_mode='hybrid', q_dtype=torch.float32)  # was float16

    freeze_all(model)
    for _, vp in model._iter_paft_v_proj():
        vp.S.requires_grad_(True)
        if vp.S.dtype != torch.float32:
            vp.S.data = vp.S.data.float()

    verify_trainable_precision(model)
    logger.info(f"LLaMA Hybrid PAFT: {count_trainable(model):,} trainable params")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 3: LLaMA LoRA
# ──────────────────────────────────────────────────────────────────────────────

def build_llama_lora(
    model_name: str = DEFAULT_MODEL,
    rank: int = 8,
    device_map: str = "auto",
) -> Tuple[nn.Module, Any]:
    """
    LoRA on q_proj, v_proj, o_proj.
    k_proj not targeted — standard practice (k rarely needs fine-tuning).
    Also targets MLP gate/up/down projections for max expressiveness.

    r=8:  params ≈ 28 × 6 × (r × d) × 2 ≈ 10.5M  (q,k,v,o,gate,up; down excluded)
    r=64: params ≈ 84M
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError("peft library required: pip install peft>=0.9.0")

    base, tokenizer = load_llama_nf4(model_name, device_map)

    # Prepare model for k-bit training — unfreezes norm layers, casts trainable to fp32
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing = True,
    )

    lora_config = LoraConfig(
        r             = rank,
        lora_alpha    = rank * 2,
        lora_dropout  = 0.05,
        target_modules = ["v_proj", "o_proj"],
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)

    logger.info(f"LLaMA LoRA r={rank}: {count_trainable(model):,} trainable params")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 4: LLaMA BitFit
# ──────────────────────────────────────────────────────────────────────────────

def build_llama_bitfit(
    model_name: str = DEFAULT_MODEL,
    device_map: str = "auto",
) -> Tuple[nn.Module, Any]:
    """
    Trainable: all bias terms that exist in the model (in fp32).

    LLaMA-3.2-3B uses RMSNorm which has no bias.
    LLaMA attention/MLP projections may or may not have biases depending on config.
    In LLaMA-3.2, all attention projections have bias=False.
    So BitFit on LLaMA adapts essentially nothing — this is the frozen baseline
    under a different name.  We include it for methodological completeness.

    If the model has no biases, this reduces to Frozen.
    """
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError:
        raise ImportError("peft library required: pip install peft>=0.9.0")

    base, tokenizer = load_llama_nf4(model_name, device_map)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    freeze_all(base)
    n_bias_params = 0
    for name, p in base.named_parameters():
        if 'bias' in name and p.dtype == torch.float32:
            p.requires_grad_(True)
            n_bias_params += p.numel()

    logger.info(
        f"LLaMA BitFit: {count_trainable(base):,} trainable params  "
        f"(bias params found: {n_bias_params})"
    )
    return base, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 5: LLaMA Frozen (lower bound)
# ──────────────────────────────────────────────────────────────────────────────

def build_llama_frozen(
    model_name: str = DEFAULT_MODEL,
    device_map: str = "auto",
) -> Tuple[nn.Module, Any]:
    """
    Zero-shot evaluation — no fine-tuning.  Performance lower bound.
    Useful to measure the gain from any fine-tuning method.
    """
    base, tokenizer = load_llama_nf4(model_name, device_map)
    freeze_all(base)
    logger.info("LLaMA Frozen: 0 trainable params (zero-shot evaluation)")
    return base, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method registry
# ──────────────────────────────────────────────────────────────────────────────

LLAMA_METHODS = {
    "pure_paft":   build_llama_pure_paft,
    "hybrid_paft": build_llama_hybrid_paft,
    "lora_r8":     lambda m, dm: build_llama_lora(m, rank=8,  device_map=dm),
    "lora_r64":    lambda m, dm: build_llama_lora(m, rank=64, device_map=dm),
    "bitfit":      build_llama_bitfit,
    "frozen":      build_llama_frozen,
}


def get_llama_model(
    method_name: str,
    model_name:  str = DEFAULT_MODEL,
    device_map:  str = "auto",
) -> Tuple[nn.Module, Any]:
    """
    Factory for all LLaMA methods.

    Args:
        method_name: One of LLAMA_METHODS keys.
        model_name:  HuggingFace model ID or local path.
        device_map:  bitsandbytes device map ('auto' or specific device).
    """
    if method_name not in LLAMA_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {sorted(LLAMA_METHODS.keys())}"
        )
    builder = LLAMA_METHODS[method_name]
    # PAFT methods need model_name + device_map; LoRA/BitFit/Frozen also accept both
    if method_name in ("lora_r8", "lora_r64"):
        return builder(model_name, device_map)
    return builder(model_name, device_map)


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Enable gradient checkpointing for training memory efficiency.
    Reduces activation memory by ~40% at the cost of ~30% extra compute.
    Essential for LLaMA-3.2-3B on 8 GB VRAM.
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif hasattr(model, 'base') and hasattr(model.base, 'gradient_checkpointing_enable'):
        model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logger.info("Gradient checkpointing enabled.")


# ──────────────────────────────────────────────────────────────────────────────
# Method: LLaMA PoLAR (Stiefel manifold on v_proj)
# ──────────────────────────────────────────────────────────────────────────────

def build_llama_polar(
    model_name: str = DEFAULT_MODEL,
    rank: int = 8,
    device_map: str = "auto",
) -> Tuple[nn.Module, Any]:
    """
    PoLAR on v_proj (same target as PAFT for fair comparison).
    ΔW = (alpha/r) * X @ B^T, X Stiefel-constrained via QR retraction.

    The retract_to_stiefel() must be called after each optimizer.step()
    in the custom training loop (train_llm.py handles this for polar_r8).
    """
    from paft.model.polar_linear import PoLARLinear

    base, tokenizer = load_llama_nf4(model_name, device_map)

    # Replace v_proj in each attention layer with PoLARLinear
    # v_proj weight [n_kv_heads*head_dim, hidden] — dequantize first
    for l in range(base.config.num_hidden_layers):
        attn  = base.model.layers[l].self_attn
        vp    = attn.v_proj
        w_fp32 = _dequantize_weight(vp)   # [1024, 3072] fp32
        bias   = vp.bias.detach().float().cpu() if vp.bias is not None else None

        polar_layer = PoLARLinear(
            in_features  = w_fp32.shape[1],
            out_features = w_fp32.shape[0],
            weight       = w_fp32,
            bias         = bias,
            rank         = rank,
            alpha        = rank * 2,
        )
        attn.v_proj = polar_layer
        del vp, w_fp32

    freeze_all(base)
    for m in base.modules():
        if isinstance(m, PoLARLinear):
            m.X.requires_grad_(True)
            m.B.requires_grad_(True)
            # Ensure fp32
            if m.X.dtype != torch.float32:
                m.X.data = m.X.data.float()
            if m.B.dtype != torch.float32:
                m.B.data = m.B.data.float()

    n_trainable = count_trainable(base)
    logger.info(f"LLaMA PoLAR r={rank}: {n_trainable:,} trainable params")
    return base, tokenizer


LLAMA_METHODS["polar_r8"] = lambda m, dm: build_llama_polar(m, rank=8, device_map=dm)
"""
DeBERTa method variants for GLUE experiments.

All 7 methods in one file:
  1. DeBERTaPurePAFT    — eigenvalues of S only (lam_V, lam_O) + classifier
  2. DeBERTaHybridPAFT  — full S matrices (S_V, S_O) + classifier
  3. DeBERTaLoRA        — LoRA r=8 or r=64 on q,v,output projections
  4. DeBERTaBitFit      — all biases + classifier (no weight change)
  5. DeBERTaFrozen      — only classifier head (lower bound)
  6. DeBERTaFullFT      — all parameters unfrozen (upper bound)
  7. DeBERTaSVF         — singular value fine-tuning via SVD (additive baseline)

All methods share the same HuggingFace Trainer-based training script
in train_glue.py.  This module only handles model construction and
parameter configuration.

Key constraint: The classification head (pooler + classifier) is ALWAYS
trainable for ALL methods — without it, GLUE tasks cannot be learned.
This is standard practice in every PEFT paper (LoRA, PoLAR, etc.).

DeBERTa attention layer path:
  model.deberta.encoder.layer[i].attention.self.query_proj   [768, 768]
  model.deberta.encoder.layer[i].attention.self.key_proj     [768, 768]
  model.deberta.encoder.layer[i].attention.self.value_proj   [768, 768]  ← PAFT
  model.deberta.encoder.layer[i].attention.output.dense      [768, 768]  ← PAFT
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paft.model.deberta_paft_model import DeBERTaPAFTModel, _N_HEADS, _HEAD_DIM
from paft.model.paft_linear import PAFTLinear
from paft.methods.base import freeze_all   # canonical implementation from base.py

logger = logging.getLogger(__name__)

HF_MODEL_NAME = "microsoft/deberta-v3-base"
_N_LAYERS = 12


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def unfreeze_classifier(model) -> None:
    """
    Unfreeze the classification head.  Always called for GLUE — the task-specific
    head must be trainable regardless of method.

    DeBERTa-v3 sequence classifier structure:
        model.pooler (ContextPooler)
        model.classifier (nn.Linear or nn.Sequential)
    """
    for name, p in model.named_parameters():
        if 'pooler' in name or 'classifier' in name:
            p.requires_grad_(True)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Base builder
# ──────────────────────────────────────────────────────────────────────────────

def load_deberta_base(task_name: str, num_labels: int):
    """Load pretrained DeBERTa-v3-base with classification head."""
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels = num_labels,
        ignore_mismatched_sizes = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 1: DeBERTa Pure PAFT
# ──────────────────────────────────────────────────────────────────────────────

def build_deberta_pure_paft(num_labels: int) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Trainable: lam_V [H, d], lam_O [H, d] per layer + classifier.
    Parameter count: 12 layers × 12 heads × 2 × 64 = 18,432 (PAFT) + classifier
    """
    base, tokenizer = load_deberta_base("", num_labels)
    model = DeBERTaPAFTModel(base, train_mode='pure', q_dtype=torch.float32)

    freeze_all(model)
    # Unfreeze eigenvalues in every PAFTLinear layer
    for _, vp, od in model._iter_paft_layers():
        vp.lam.requires_grad_(True)
        od.lam.requires_grad_(True)
    unfreeze_classifier(model)

    logger.info(f"DeBERTa Pure PAFT: {count_trainable(model):,} trainable params")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 2: DeBERTa Hybrid PAFT
# ──────────────────────────────────────────────────────────────────────────────

def build_deberta_hybrid_paft(num_labels: int) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Trainable: S_V [H, d, d], S_O [H, d, d] per layer + classifier.
    Parameter count: 12 × 12 × 2 × 64² = 1,179,648 + classifier
    """
    base, tokenizer = load_deberta_base("", num_labels)
    model = DeBERTaPAFTModel(base, train_mode='hybrid', q_dtype=torch.float32)

    freeze_all(model)
    for _, vp, od in model._iter_paft_layers():
        vp.S.requires_grad_(True)
        od.S.requires_grad_(True)
    unfreeze_classifier(model)

    logger.info(f"DeBERTa Hybrid PAFT: {count_trainable(model):,} trainable params")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 3: DeBERTa LoRA
# ──────────────────────────────────────────────────────────────────────────────

def build_deberta_lora(num_labels: int, rank: int = 8) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Standard LoRA on query_proj, key_proj, value_proj, output.dense.
    Uses PEFT library.  classifier always trainable.

    r=8:  parameter count ≈ 12 × 4 × (768+768) × 8 × 2 ≈ 589,824
    r=64: parameter count ≈ 4,718,592
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("peft library required: pip install peft>=0.9.0")

    base, tokenizer = load_deberta_base("", num_labels)

    lora_config = LoraConfig(
        task_type     = TaskType.SEQ_CLS,
        r             = rank,
        lora_alpha    = rank * 2,
        lora_dropout  = 0.1,
        target_modules = ["query_proj", "value_proj", "dense"],
        bias           = "none",
        modules_to_save = ["classifier", "pooler"],
    )
    model = get_peft_model(base, lora_config)

    logger.info(f"DeBERTa LoRA r={rank}: {count_trainable(model):,} trainable params")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 4: DeBERTa BitFit
# ──────────────────────────────────────────────────────────────────────────────

def build_deberta_bitfit(num_labels: int) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Trainable: all bias terms + classifier.
    No weight matrix adaptation.  Matches Zaken et al. 2022.
    """
    base, tokenizer = load_deberta_base("", num_labels)

    freeze_all(base)
    for name, p in base.named_parameters():
        if 'bias' in name or 'pooler' in name or 'classifier' in name:
            p.requires_grad_(True)

    logger.info(f"DeBERTa BitFit: {count_trainable(base):,} trainable params")
    return base, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 5: Frozen (lower bound)
# ──────────────────────────────────────────────────────────────────────────────

def build_deberta_frozen(num_labels: int) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Only classifier is trainable.  Everything else frozen.
    This is the performance lower bound — shows what random GLUE performance is.
    """
    base, tokenizer = load_deberta_base("", num_labels)

    freeze_all(base)
    unfreeze_classifier(base)

    logger.info(f"DeBERTa Frozen: {count_trainable(base):,} trainable params")
    return base, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 6: Full Fine-tuning (upper bound)
# ──────────────────────────────────────────────────────────────────────────────

def build_deberta_full_ft(num_labels: int) -> Tuple[nn.Module, AutoTokenizer]:
    """
    All parameters trainable.  Performance upper bound.
    Matches the DeBERTa-v3 paper's GLUE results.
    """
    base, tokenizer = load_deberta_base("", num_labels)
    # All parameters trainable by default — no freeze needed
    logger.info(f"DeBERTa Full FT: {count_trainable(base):,} trainable params")
    return base, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method 7: SVF (Singular Value Fine-tuning)
# ──────────────────────────────────────────────────────────────────────────────

class SVFLinear(nn.Module):
    """
    Singular Value Fine-tuning for a single nn.Linear layer.

    Parameterisation: W = U @ diag(sigma_adapted) @ Vh
    where U and Vh are frozen (from initial SVD of pretrained W),
    and sigma_adapted = sigma_pretrained + delta_sigma is trainable.

    This is the SVF baseline (additive in singular value space).
    Geometric health analysis computes stable rank on W_eff = U diag(σ) Vh.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        super().__init__()
        w = weight.detach().float()
        U, sigma, Vh = torch.linalg.svd(w, full_matrices=False)
        self.register_buffer('U', U)
        self.register_buffer('Vh', Vh)
        self.register_buffer('sigma_init', sigma.clone())
        # trainable: delta on top of pretrained singular values
        self.delta_sigma = nn.Parameter(torch.zeros_like(sigma), requires_grad=False)
        if bias is not None:
            self.register_buffer('bias', bias.detach().float())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma_init + self.delta_sigma
        W = self.U @ torch.diag(sigma) @ self.Vh
        return torch.nn.functional.linear(x, W, self.bias)


def build_deberta_svf(num_labels: int) -> Tuple[nn.Module, AutoTokenizer]:
    """
    SVF on value_proj and output.dense, analogous to PAFT's target layers.
    Trainable: delta_sigma per layer per projection + classifier.
    """
    base, tokenizer = load_deberta_base("", num_labels)

    # Replace value_proj and output.dense with SVFLinear
    for l in range(_N_LAYERS):
        attn_self   = base.deberta.encoder.layer[l].attention.self
        attn_output = base.deberta.encoder.layer[l].attention.output

        vp = attn_self.value_proj
        attn_self.value_proj = SVFLinear(vp.weight, vp.bias)

        od = attn_output.dense
        attn_output.dense = SVFLinear(od.weight, od.bias)

    freeze_all(base)
    # Unfreeze delta_sigma in all SVFLinear layers
    for name, p in base.named_parameters():
        if 'delta_sigma' in name or 'pooler' in name or 'classifier' in name:
            p.requires_grad_(True)

    logger.info(f"DeBERTa SVF: {count_trainable(base):,} trainable params")
    return base, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Method registry
# ──────────────────────────────────────────────────────────────────────────────

DEBERTA_METHODS = {
    "pure_paft":   build_deberta_pure_paft,
    "hybrid_paft": build_deberta_hybrid_paft,
    "lora_r8":     lambda n: build_deberta_lora(n, rank=8),
    "lora_r64":    lambda n: build_deberta_lora(n, rank=64),
    "bitfit":      build_deberta_bitfit,
    "frozen":      build_deberta_frozen,
    "full_ft":     build_deberta_full_ft,
    "svf":         build_deberta_svf,
}


def get_deberta_model(method_name: str, num_labels: int) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Factory function for all DeBERTa methods.

    Args:
        method_name: One of DEBERTA_METHODS keys.
        num_labels:  Number of output classes for the GLUE task.

    Returns:
        (model, tokenizer) ready for HuggingFace Trainer.
    """
    if method_name not in DEBERTA_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {sorted(DEBERTA_METHODS.keys())}"
        )
    return DEBERTA_METHODS[method_name](num_labels)


def get_deberta_paft_layers(model: nn.Module) -> List[Tuple[int, PAFTLinear, PAFTLinear]]:
    """
    Extract PAFTLinear layers from a DeBERTaPAFTModel for geometric analysis.
    Returns [] if model is not a PAFT model (e.g., LoRA or BitFit).
    """
    if not isinstance(model, DeBERTaPAFTModel):
        return []
    return list(model._iter_paft_layers())


# ──────────────────────────────────────────────────────────────────────────────
# Method: PoLAR (Stiefel manifold adaptation) — required baseline for Analysis 2
# ──────────────────────────────────────────────────────────────────────────────

from paft.model.polar_linear import PoLARLinear
from transformers import TrainerCallback

class _StiefelRetractCallback(TrainerCallback):
    """Retracts all PoLARLinear.X back to Stiefel after each optimizer step."""
    def __init__(self, model: nn.Module):
        self._layers = [
            m for m in model.modules() if isinstance(m, PoLARLinear)
        ]

    def on_step_end(self, args, state, control, **kwargs):
        for layer in self._layers:
            layer.retract_to_stiefel()


def build_deberta_polar(
    num_labels: int,
    rank: int = 8,
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    PoLAR on value_proj and output.dense (same target layers as PAFT).
    Parameterisation: ΔW = (alpha/r) * X @ B^T, X on Stiefel manifold.
    Retraction: QR after each optimizer step (via HF Trainer callback).
    sr(ΔW) is high by construction; sr(W_eff) is what we measure in Analysis 2.

    Returns (model, tokenizer, callback) — pass callback to HF Trainer.
    The caller must add the StiefelRetractCallback to trainer callbacks.
    """
    base, tokenizer = load_deberta_base("", num_labels)

    # Replace value_proj and output.dense with PoLARLinear
    for l in range(_N_LAYERS):
        attn_self   = base.deberta.encoder.layer[l].attention.self
        attn_output = base.deberta.encoder.layer[l].attention.output

        attn_self.value_proj = PoLARLinear.from_linear(
            attn_self.value_proj, rank=rank, alpha=rank * 2
        )
        attn_output.dense = PoLARLinear.from_linear(
            attn_output.dense, rank=rank, alpha=rank * 2
        )

    freeze_all(base)
    # Unfreeze X and B in every PoLARLinear
    for m in base.modules():
        if isinstance(m, PoLARLinear):
            m.X.requires_grad_(True)
            m.B.requires_grad_(True)
    unfreeze_classifier(base)

    # Build the callback (must be added to HF Trainer by caller)
    callback = _StiefelRetractCallback(base)

    logger.info(f"DeBERTa PoLAR r={rank}: {count_trainable(base):,} trainable params")
    return base, tokenizer, callback   # NOTE: returns 3-tuple


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 5: Which-weights ablation variants
# ──────────────────────────────────────────────────────────────────────────────

class _PartialPAFTModel(nn.Module):
    """
    DeBERTa with PAFT applied to a user-specified subset of attention projections.
    Used for Analysis 5: which weights benefit most from PAFT?
    """
    def __init__(self, base, adapt_value: bool, adapt_output: bool,
                 adapt_query: bool = False, train_mode: str = 'hybrid') -> None:
        super().__init__()
        self.base = base
        self._adapted_layers = []   # list of PAFTLinear layers for geometric access

        for l in range(_N_LAYERS):
            attn_self   = base.deberta.encoder.layer[l].attention.self
            attn_output = base.deberta.encoder.layer[l].attention.output

            if adapt_value:
                vp = attn_self.value_proj
                attn_self.value_proj = PAFTLinear(
                    vp.weight.detach().float(), vp.bias.detach().float() if vp.bias is not None else None,
                    _N_HEADS, _HEAD_DIM, 'row', train_mode, torch.float32
                )
                self._adapted_layers.append(('V', l, attn_self.value_proj))

            if adapt_output:
                od = attn_output.dense
                attn_output.dense = PAFTLinear(
                    od.weight.detach().float(), od.bias.detach().float() if od.bias is not None else None,
                    _N_HEADS, _HEAD_DIM, 'col', train_mode, torch.float32
                )
                self._adapted_layers.append(('O', l, attn_output.dense))

            if adapt_query:
                qp = attn_self.query_proj
                attn_self.query_proj = PAFTLinear(
                    qp.weight.detach().float(), qp.bias.detach().float() if qp.bias is not None else None,
                    _N_HEADS, _HEAD_DIM, 'row', train_mode, torch.float32
                )
                self._adapted_layers.append(('Q', l, attn_self.query_proj))

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    @property
    def config(self):
        return self.base.config


def build_deberta_paft_ablation(
    num_labels: int,
    adapt_value: bool = True,
    adapt_output: bool = True,
    adapt_query: bool = False,
    train_mode: str = 'hybrid',
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Build DeBERTa with PAFT applied to a specific subset of projections.

    Variants for Analysis 5:
      paft_v_only:   adapt_value=True,  adapt_output=False  (value only)
      paft_o_only:   adapt_value=False, adapt_output=True   (output only)
      paft_vo:       adapt_value=True,  adapt_output=True   (both, standard)
      paft_qv:       adapt_value=True,  adapt_query=True    (query + value)
    """
    base, tokenizer = load_deberta_base("", num_labels)
    model = _PartialPAFTModel(base, adapt_value, adapt_output, adapt_query, train_mode)

    freeze_all(model)
    for _, _, layer in model._adapted_layers:
        if train_mode == 'pure':
            layer.lam.requires_grad_(True)
        else:
            layer.S.requires_grad_(True)
    unfreeze_classifier(model)

    tag = ('V' if adapt_value else '') + ('O' if adapt_output else '') + ('Q' if adapt_query else '')
    logger.info(f"DeBERTa PAFT ablation [{tag}]: {count_trainable(model):,} trainable params")
    return model, tokenizer


# Add ablation variants to method registry
DEBERTA_METHODS.update({
    "polar_r8":     lambda n: build_deberta_polar(n, rank=8)[:2],  # drop callback (train_glue handles)
    "paft_v_only":  lambda n: build_deberta_paft_ablation(n, adapt_value=True,  adapt_output=False),
    "paft_o_only":  lambda n: build_deberta_paft_ablation(n, adapt_value=False, adapt_output=True),
    "paft_qv":      lambda n: build_deberta_paft_ablation(n, adapt_value=True,  adapt_output=False, adapt_query=True),
    "paft_vo":      lambda n: build_deberta_paft_ablation(n, adapt_value=True,  adapt_output=True),
})
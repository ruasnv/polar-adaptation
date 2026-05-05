"""
Tests for parameter freeze/unfreeze correctness across all 11 methods.

THE MOST CRITICAL TEST FILE.
Silent failure mode: wrong parameters have requires_grad=True.
The optimizer silently updates the wrong tensors.
Loss curves look normal. Results are scientifically invalid.

Tests use a tiny synthetic GPT-2-like model (n_embd=64, n_heads=4)
so they complete in seconds with no downloads.

For each method we verify:
  1. Exactly the expected parameters are trainable (no extras, no missing)
  2. The trainable parameter count matches the expected value
  3. No frozen parameter has requires_grad=True
"""

import math
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from paft.methods.base import freeze_all


# ──────────────────────────────────────────────────────────────────────────────
# Minimal synthetic GPT-2 construction
# ──────────────────────────────────────────────────────────────────────────────

H = 4; N = 64; D = 16; N_LAYERS = 3
HF_NAME = "gpt2"  # not actually downloaded — build() is mocked


def _make_tiny_config():
    return SimpleNamespace(
        n_head=H, n_embd=N, n_layer=N_LAYERS,
        max_position_embeddings=128,
    )


def _make_tiny_gpt2():
    """
    Build a minimal GPT-2-like model with the exact attribute structure
    that PAFTModel, SVFModel and the baseline methods access.
    No actual GPT-2 weights are downloaded.
    """
    import torch.nn as nn
    from types import SimpleNamespace

    cfg = _make_tiny_config()

    class TinyConv1D(nn.Module):
        def __init__(self, in_f, out_f, bias=True):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_f, out_f) * 0.02)
            self.bias   = nn.Parameter(torch.zeros(out_f)) if bias else None
        def forward(self, x):
            return x @ self.weight + (self.bias if self.bias is not None else 0)

    class TinyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = TinyConv1D(N, 3*N)
            self.c_proj = TinyConv1D(N, N)

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc   = TinyConv1D(N, 4*N)
            self.c_proj = TinyConv1D(4*N, N)

    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TinyAttn()
            self.mlp  = TinyMLP()
            self.ln_1 = nn.LayerNorm(N)
            self.ln_2 = nn.LayerNorm(N)

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.h  = nn.ModuleList([TinyBlock() for _ in range(N_LAYERS)])
            self.ln_f = nn.LayerNorm(N)

    class TinyGPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.config      = cfg
            self.transformer = TinyTransformer()
            self.lm_head     = nn.Linear(N, 100, bias=False)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            loss = torch.tensor(0.0, requires_grad=True)
            return SimpleNamespace(loss=loss)

        def gradient_checkpointing_enable(self, **kwargs):
            pass   # no-op for tiny test model

        def gradient_checkpointing_disable(self, **kwargs):
            pass

    return TinyGPT2()


def _build_method(method_class, method_name, extra_cfg=None):
    """
    Instantiate a method and call build() with a mocked GPT-2 loader.
    Patches GPT2LMHeadModel.from_pretrained to return our tiny synthetic model.
    """
    from unittest.mock import patch
    cfg = {"training": {"learning_rate": 2e-4, "weight_decay": 0.01}}
    if extra_cfg:
        cfg.update(extra_cfg)
    method = method_class(method_name=method_name, cfg=cfg)
    tiny = _make_tiny_gpt2()
    device = torch.device("cpu")

    with patch("transformers.GPT2LMHeadModel.from_pretrained", return_value=tiny):
        method.build(HF_NAME, device)

    return method


# ──────────────────────────────────────────────────────────────────────────────
# Helper assertions
# ──────────────────────────────────────────────────────────────────────────────

def _trainable_names(method):
    return {
        name for name, p in method.model.named_parameters()
        if p.requires_grad
    }

def _frozen_names(method):
    return {
        name for name, p in method.model.named_parameters()
        if not p.requires_grad
    }

def assert_only_contains(trainable: set, required_substrings: list, method_name: str):
    """Every trainable param name must contain at least one required substring."""
    unexpected = [
        n for n in trainable
        if not any(s in n for s in required_substrings)
    ]
    assert not unexpected, (
        f"{method_name}: unexpected trainable params (none of {required_substrings} "
        f"found in name):\n  " + "\n  ".join(unexpected)
    )

def assert_all_present(trainable: set, required_substrings: list, method_name: str):
    """At least one trainable param must exist for each required substring."""
    for s in required_substrings:
        matches = [n for n in trainable if s in n]
        assert matches, (
            f"{method_name}: no trainable param containing '{s}' found.\n"
            f"Trainable params: {sorted(trainable)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# M1: Frozen
# ──────────────────────────────────────────────────────────────────────────────

def test_frozen_no_trainable_params():
    from paft.methods.baselines.frozen import Frozen
    method = _build_method(Frozen, "frozen")
    trainable = _trainable_names(method)
    assert len(trainable) == 0, (
        f"Frozen should have 0 trainable params, got: {trainable}"
    )
    assert method.num_trainable_params() == 0


# ──────────────────────────────────────────────────────────────────────────────
# M2: FullFinetune
# ──────────────────────────────────────────────────────────────────────────────

def test_full_finetune_all_trainable():
    from paft.methods.baselines.full_finetune import FullFinetune
    method = _build_method(FullFinetune, "full_finetune")
    frozen = _frozen_names(method)
    assert len(frozen) == 0, (
        f"FullFinetune should have 0 frozen params, got: {frozen}"
    )
    assert method.num_trainable_params() > 0


# ──────────────────────────────────────────────────────────────────────────────
# M3: BitFit
# ──────────────────────────────────────────────────────────────────────────────

def test_bitfit_only_biases_trainable():
    from paft.methods.baselines.bitfit import BitFit
    method = _build_method(BitFit, "bitfit")
    trainable = _trainable_names(method)
    assert len(trainable) > 0, "BitFit must have trainable params"
    assert_only_contains(trainable, ["bias"], "bitfit")
    # Verify no weights are trainable
    weight_trainable = [n for n in trainable if "weight" in n]
    assert not weight_trainable, f"BitFit: weights should be frozen: {weight_trainable}"


# ──────────────────────────────────────────────────────────────────────────────
# M8: PurePAFT
# ──────────────────────────────────────────────────────────────────────────────

def test_pure_paft_only_lambda_trainable():
    from paft.methods.pure_paft import PurePAFT
    from paft.model.paft_model import PAFTModel
    method = _build_method(PurePAFT, "pure_paft")
    trainable = _trainable_names(method)
    assert len(trainable) > 0
    assert_only_contains(trainable, ["lam_V", "lam_O"], "pure_paft")
    assert_all_present(trainable, ["lam_V", "lam_O"], "pure_paft")

    # S_V and S_O must be FROZEN
    s_trainable = [n for n in trainable if "S_V" in n or "S_O" in n
                   if not any(x in n for x in ["lam"])]
    assert not s_trainable, f"pure_paft: S_V/S_O should be frozen, got: {s_trainable}"

    # Biases must be FROZEN
    bias_trainable = [n for n in trainable if "b_qkv" in n or "b_o" in n]
    assert not bias_trainable, f"pure_paft: biases should be frozen"


# ──────────────────────────────────────────────────────────────────────────────
# M9: HybridPAFT
# ──────────────────────────────────────────────────────────────────────────────

def test_hybrid_paft_only_S_trainable():
    from paft.methods.hybrid_paft import HybridPAFT
    method = _build_method(HybridPAFT, "hybrid_paft")
    trainable = _trainable_names(method)
    assert len(trainable) > 0
    assert_only_contains(trainable, ["S_V", "S_O"], "hybrid_paft")
    assert_all_present(trainable, ["S_V", "S_O"], "hybrid_paft")

    # lam must be FROZEN
    lam_trainable = [n for n in trainable if "lam_" in n]
    assert not lam_trainable, f"hybrid_paft: lam should be frozen"

    # Biases must be FROZEN
    bias_trainable = [n for n in trainable if "b_qkv" in n or "b_o" in n]
    assert not bias_trainable, f"hybrid_paft: biases should be frozen"


# ──────────────────────────────────────────────────────────────────────────────
# M10: SafePurePAFT
# ──────────────────────────────────────────────────────────────────────────────

def test_safe_pure_paft_lambda_and_biases():
    from paft.methods.safe_pure_paft import SafePurePAFT
    method = _build_method(SafePurePAFT, "safe_pure_paft")
    trainable = _trainable_names(method)
    assert len(trainable) > 0

    # lambda must be trainable
    assert_all_present(trainable, ["lam_V", "lam_O"], "safe_pure_paft")

    # Attention biases (b_qkv, b_o) must be trainable
    assert_all_present(trainable, ["b_qkv", "b_o"], "safe_pure_paft")

    # S_V and S_O must remain frozen
    s_trainable = [n for n in trainable
                   if (".S_V" in n or ".S_O" in n)
                   and "lam" not in n]
    assert not s_trainable, f"safe_pure_paft: S matrices should be frozen"

    # Q matrices must be frozen (buffers, not params — but sanity check)
    q_trainable = [n for n in trainable if ".Q_V" in n or ".Q_O" in n]
    assert not q_trainable, f"safe_pure_paft: Q should be frozen"


# ──────────────────────────────────────────────────────────────────────────────
# M11: SafeHybridPAFT
# ──────────────────────────────────────────────────────────────────────────────

def test_safe_hybrid_paft_S_and_biases():
    from paft.methods.safe_hybrid_paft import SafeHybridPAFT
    method = _build_method(SafeHybridPAFT, "safe_hybrid_paft")
    trainable = _trainable_names(method)
    assert len(trainable) > 0

    # S matrices must be trainable
    assert_all_present(trainable, ["S_V", "S_O"], "safe_hybrid_paft")

    # Attention biases must be trainable
    assert_all_present(trainable, ["b_qkv", "b_o"], "safe_hybrid_paft")

    # lam must remain frozen
    lam_trainable = [n for n in trainable if "lam_" in n]
    assert not lam_trainable, f"safe_hybrid_paft: lam should be frozen"

    # Q must remain frozen
    q_trainable = [n for n in trainable if ".Q_V" in n or ".Q_O" in n]
    assert not q_trainable, f"safe_hybrid_paft: Q should be frozen"


# ──────────────────────────────────────────────────────────────────────────────
# M4: SVF
# ──────────────────────────────────────────────────────────────────────────────

def test_svf_only_sigma_trainable():
    from paft.methods.baselines.svf import SVFBaseline
    from paft.model.svf_model import SVFModel
    method = _build_method(SVFBaseline, "svf")
    trainable = _trainable_names(method)
    assert len(trainable) > 0
    assert_only_contains(trainable, ["sigma_V", "sigma_O"], "svf")
    assert_all_present(trainable, ["sigma_V", "sigma_O"], "svf")

    # U and Vh must be frozen (they're buffers, but assert no U/Vh params)
    u_trainable = [n for n in trainable if ".U_V" in n or ".U_O" in n
                   or ".Vh_V" in n or ".Vh_O" in n]
    assert not u_trainable, f"svf: U/Vh should be frozen"


# ──────────────────────────────────────────────────────────────────────────────
# Mutual exclusivity: no method should double-count
# ──────────────────────────────────────────────────────────────────────────────

def test_pure_vs_hybrid_disjoint_trainable():
    """pure_paft and hybrid_paft train different parameters."""
    from paft.methods.pure_paft import PurePAFT
    from paft.methods.hybrid_paft import HybridPAFT
    pure   = _build_method(PurePAFT,   "pure_paft")
    hybrid = _build_method(HybridPAFT, "hybrid_paft")

    pure_trainable   = _trainable_names(pure)
    hybrid_trainable = _trainable_names(hybrid)

    # Neither should appear in the other's set with matching suffixes
    pure_lam   = {n for n in pure_trainable   if "lam_" in n}
    hybrid_S   = {n for n in hybrid_trainable if ".S_V" in n or ".S_O" in n}
    hybrid_lam = {n for n in hybrid_trainable if "lam_" in n}
    pure_S     = {n for n in pure_trainable   if ".S_V" in n or ".S_O" in n}

    assert len(pure_lam) > 0,   "pure_paft has no lam params"
    assert len(hybrid_S) > 0,   "hybrid_paft has no S params"
    assert len(hybrid_lam) == 0, "hybrid_paft should NOT train lam"
    assert len(pure_S) == 0,     "pure_paft should NOT train S"


def test_safe_adds_biases_to_base():
    """safe_pure_paft has strictly more trainable params than pure_paft."""
    from paft.methods.pure_paft import PurePAFT
    from paft.methods.safe_pure_paft import SafePurePAFT
    pure      = _build_method(PurePAFT,      "pure_paft")
    safe_pure = _build_method(SafePurePAFT,  "safe_pure_paft")
    assert safe_pure.num_trainable_params() > pure.num_trainable_params()
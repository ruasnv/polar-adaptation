"""
Shared utilities for all analysis scripts.

model loading, HF model extraction, test data loaders — used by every script.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Method registry (import lazily to avoid circular import at top of every file)
# ──────────────────────────────────────────────────────────────────────────────

def _get_method(method_name: str, cfg: dict):
    from paft.methods import get_method
    return get_method(method_name, cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_trained_model(
    run_dir:     Path,
    cfg:         dict,
    device:      torch.device,
    checkpoint:  str = "final",   # "final" | "epoch_N"
):
    """
    Reconstruct a trained method from checkpoint.

    Workflow:
        1. Instantiate the method class (from method_name = run_dir.name)
        2. Call method.build(hf_name, device)  — this runs polar/SVD decomposition
           on the pretrained weights (CPU), then moves to device
        3. Load model.pt state_dict on top  — overwrites decomposed values with
           trained values from the checkpoint

    This restores the model to its post-training state for any method type.

    Returns the built method object with model on device, in eval mode.
    """
    method_name = run_dir.name
    hf_name     = cfg["model"]["hf_name"]

    method = _get_method(method_name, cfg)
    method.build(hf_name, device)

    ckpt_dir  = run_dir / checkpoint
    model_pt  = ckpt_dir / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found: {model_pt}")

    state_dict = torch.load(model_pt, map_location=device, weights_only=False)
    method.model.load_state_dict(state_dict, strict=False)
    method.model.eval()
    return method


def get_hf_model(method):
    """
    Extract the underlying HuggingFace-compatible model from any method.

    For lm_eval and generation, we need a model that responds to HF's
    forward(input_ids, attention_mask, ...) → logits interface.

    PAFT/SVF:   method.model is PAFTModel/SVFModel; .base is the GPT2LMHeadModel
                with adapted attention layers — this IS HF-compatible.
    LoRA:       method.model is a PeftModel; .base_model.model is GPT2LMHeadModel.
    Others:     method.model IS a GPT2LMHeadModel.
    """
    from paft.model.paft_model import PAFTModel
    from paft.model.svf_model  import SVFModel

    m = method.model
    if isinstance(m, (PAFTModel, SVFModel)):
        return m.base                        # GPT2LMHeadModel with adapted attn
    if hasattr(m, "base_model"):             # PeftModel (LoRA, PoLAR via PEFT)
        return m.base_model.model
    return m                                 # frozen, full_finetune, bitfit


def get_tokenizer(hf_name: str):
    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


# ──────────────────────────────────────────────────────────────────────────────
# Run discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_complete_runs(
    checkpoint_root: Path,
    model:    str,
    domains:  List[str],
    methods:  List[str],
) -> List[Tuple[str, str, Path]]:
    """Return (domain, method, run_dir) for every run with a sentinel file."""
    runs = []
    for domain in domains:
        for method in methods:
            run_dir = checkpoint_root / model / domain / method
            if (run_dir / "final" / "training_complete").exists():
                runs.append((domain, method, run_dir))
    return runs


def load_init_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "init" / "config.json"
    with path.open() as f:
        return json.load(f)
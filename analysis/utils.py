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

# Long-form metric keys matching values written to metrics.json.
# Must match build_cache.py's TASK_PRIMARY exactly.
TASK_PRIMARY = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp":  "f1",
    "rte":  "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
}

def load_adapted_weights(run_dir: Path | str, tag: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Loads adapted_weights.pt from a specific tag checkpoint.
    Flattens each layer's head dimensions [H, n, d] -> [H*n, d] for 2D matrix analysis.
    Returns: (W_V_layers, W_O_layers) as lists of 2D tensors.
    """
    path = Path(run_dir) / tag / "adapted_weights.pt"
    if not path.exists():
        return [], []

    ckpt = torch.load(path, map_location="cpu")

    W_V_layers = []
    if "W_V" in ckpt:
        for W in ckpt["W_V"]:
            # Flatten [H, n, d] -> [H*n, d]
            W_V_layers.append(W.reshape(-1, W.shape[-1]))

    W_O_layers = []
    if "W_O" in ckpt:
        for W in ckpt["W_O"]:
            # Flatten [H, d, n] -> [H*d, n]
            W_O_layers.append(W.reshape(-1, W.shape[-1]))

    return W_V_layers, W_O_layers
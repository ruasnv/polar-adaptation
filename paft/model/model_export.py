"""
paft/model/model_export.py

Exports any fine-tuned model (PAFT, PoLAR, LoRA, BitFit) to a standard
HuggingFace model directory so lm-evaluation-harness can evaluate it.

This solves the eval protocol issue:
  - We TRAIN with our framework
  - We EVALUATE with lm-evaluation-harness (same tool as PoLAR)
  - Numbers are directly comparable to PoLAR's published results

For each method:
  PAFT:   Replace PAFTLinear with nn.Linear(W_eff) and save
  PoLAR:  Replace PoLARLinear with nn.Linear(W_0 + ΔW) and save
  LoRA:   Call model.merge_and_unload() then save (PEFT handles this)
  BitFit/Frozen: Just save as-is (biases/weights are standard)

After export, run lm-evaluation-harness:
    lm_eval --model hf \
            --model_args pretrained={export_path} \
            --tasks boolq,piqa,siqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
            --device cuda:0 \
            --batch_size 8 \
            --output_path {results_path}

Usage:
    python -m paft.model.model_export \
        --checkpoint_dir results/commonsense/boolq/pure_paft \
        --method pure_paft \
        --model_name meta-llama/Llama-3.2-3B \
        --output_dir exported_models/boolq/pure_paft
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Core export functions ─────────────────────────────────────────────────────

def export_llama_paft(
    model,
    output_dir: str | Path,
    tokenizer=None,
) -> None:
    """
    Replace all PAFTLinear v_proj layers with their reconstructed nn.Linear
    equivalents, then save as a standard LlamaForCausalLM.

    The resulting model is numerically identical to the trained PAFT model
    but has no PAFTLinear modules — lm-evaluation-harness can load it normally.
    """
    from paft.model.llama_paft_model import LLaMAPAFTModel
    from paft.model.paft_linear import PAFTLinear

    if not isinstance(model, LLaMAPAFTModel):
        raise ValueError("Expected LLaMAPAFTModel")

    base_model = model.base
    n_layers   = base_model.config.num_hidden_layers

    logger.info(f"Exporting LLaMA PAFT model to {output_dir} ...")
    with torch.no_grad():
        for l in range(n_layers):
            attn = base_model.model.layers[l].self_attn
            vp   = attn.v_proj
            if not isinstance(vp, PAFTLinear):
                continue

            # Reconstruct effective weight
            W_eff = vp.reconstruct_weight().float()   # [H_kv*d, hidden]
            bias  = vp.bias.float() if vp.bias is not None else None

            # Replace with standard nn.Linear
            new_linear = nn.Linear(W_eff.shape[1], W_eff.shape[0], bias=(bias is not None))
            new_linear.weight.data = W_eff
            if bias is not None:
                new_linear.bias.data = bias
            attn.v_proj = new_linear

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(output_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    logger.info(f"Exported to {output_path}")


def export_llama_polar(
    model,
    output_dir: str | Path,
    tokenizer=None,
) -> None:
    """
    Replace PoLARLinear v_proj with reconstructed nn.Linear (W_0 + ΔW).
    """
    from paft.model.polar_linear import PoLARLinear

    base_model = getattr(model, 'base', model)
    n_layers   = base_model.config.num_hidden_layers

    logger.info(f"Exporting LLaMA PoLAR model to {output_dir} ...")
    with torch.no_grad():
        for l in range(n_layers):
            attn = base_model.model.layers[l].self_attn
            vp   = attn.v_proj
            if not isinstance(vp, PoLARLinear):
                continue

            W_eff = vp.get_effective_W().float()
            bias  = vp.bias_0.float() if vp.bias_0 is not None else None

            new_linear = nn.Linear(W_eff.shape[1], W_eff.shape[0], bias=(bias is not None))
            new_linear.weight.data = W_eff
            if bias is not None:
                new_linear.bias.data = bias
            attn.v_proj = new_linear

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(output_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    logger.info(f"Exported to {output_path}")


def export_llama_lora(
    model,
    output_dir: str | Path,
    tokenizer=None,
) -> None:
    """
    Merge LoRA adapter weights into base model and save.
    Uses PEFT's built-in merge_and_unload() which handles all the math.
    """
    logger.info(f"Merging LoRA and exporting to {output_dir} ...")
    merged = model.merge_and_unload()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    logger.info(f"Exported to {output_path}")


def export_llama_standard(
    model,
    output_dir: str | Path,
    tokenizer=None,
) -> None:
    """
    For BitFit and Frozen: the base NF4 model is already a standard LlamaForCausalLM.
    Just save it (weights may include updated biases for BitFit).
    """
    base_model = getattr(model, 'base', model)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(output_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    logger.info(f"Exported to {output_path}")


# ── Auto-dispatch ─────────────────────────────────────────────────────────────

def export_model(
    model,
    method_name:  str,
    output_dir:   str | Path,
    tokenizer=None,
) -> None:
    """
    Dispatch to the correct export function based on method name.
    Call this after training completes, before running lm-evaluation-harness.
    """
    method_lower = method_name.lower()

    if 'pure_paft' in method_lower or 'hybrid_paft' in method_lower:
        export_llama_paft(model, output_dir, tokenizer)
    elif 'polar' in method_lower:
        export_llama_polar(model, output_dir, tokenizer)
    elif 'lora' in method_lower:
        export_llama_lora(model, output_dir, tokenizer)
    else:
        # BitFit, Frozen, full_ft — save standard
        export_llama_standard(model, output_dir, tokenizer)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    """
    Load a trained checkpoint and export it for lm-evaluation-harness.

    Example:
        python -m paft.model.model_export \
            --checkpoint_dir results/commonsense/boolq/pure_paft \
            --method pure_paft \
            --model_name meta-llama/Llama-3.2-3B \
            --output_dir exported_models/commonsense/boolq/pure_paft
    """
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--method",         required=True)
    p.add_argument("--model_name",     default="meta-llama/Llama-3.2-3B")
    p.add_argument("--output_dir",     required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    from paft.methods.llama_methods import get_llama_model
    import json

    # Load config to get model name
    cfg_path = Path(args.checkpoint_dir) / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", args.model_name)
    else:
        model_name = args.model_name

    logger.info(f"Loading {args.method} from {model_name} ...")
    model, tokenizer = get_llama_model(args.method, model_name)

    # Load trained adapter weights
    adapter_path = Path(args.checkpoint_dir) / "adapter_final.pt"
    if adapter_path.exists():
        logger.info(f"Loading adapter from {adapter_path} ...")
        state = torch.load(adapter_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded adapter: {len(state)} tensors, "
                    f"missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        logger.warning(f"No adapter_final.pt found at {adapter_path} — exporting base model")

    export_model(model, args.method, args.output_dir, tokenizer)


if __name__ == "__main__":
    main()
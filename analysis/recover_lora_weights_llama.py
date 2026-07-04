#!/usr/bin/env python3
"""
analysis/recover_lora_weights_llama.py

Recovers correct W_eff for LLaMA LoRA runs by loading the saved adapter
weights, merging them into the NF4 base model, extracting per-head v_proj
tensors, and computing geometric health metrics.

Mirrors analysis/recover_lora_weights.py for DeBERTa but targets the
LLaMA-3.2-3B architecture (28 layers, 8 KV heads, head_dim=128).

Usage:
    python3 -m analysis.recover_lora_weights_llama --results_dir results/llama
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
)

# LLaMA-3.2-3B architecture constants
_N_LAYERS    = 28
_N_KV_HEADS  = 8
_N_Q_HEADS   = 24
_HEAD_DIM    = 128
_HID_SIZE    = 3072

LORA_METHODS = {"lora_r8", "lora_r64"}
LORA_RANKS   = {"lora_r8": 8, "lora_r64": 64}
TASKS        = ["boolq", "hellaswag", "arc_challenge"]
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"


# ── Weight extraction ──────────────────────────────────────────────────────────

def _dequantize(layer) -> torch.Tensor:
    """Dequantize NF4 weight to fp32 [out, in]."""
    w = layer.weight
    if hasattr(w, "quant_state"):
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(
            w.data, quant_state=w.quant_state, quant_type="nf4"
        ).detach().float().cpu()
    return w.detach().float().cpu()


def _extract_llama_v_proj(merged_model) -> dict:
    """
    Extract merged v_proj weights per layer.
    Returns W_V: List[28] of Tensor[8, 3072, 128]  (H_kv, hidden, head_dim)
    """
    W_V_layers = []
    with torch.no_grad():
        for l in range(_N_LAYERS):
            vp = merged_model.model.layers[l].self_attn.v_proj
            w  = _dequantize(vp)                    # [1024, 3072]
            W_V = (w.reshape(_N_KV_HEADS, _HEAD_DIM, _HID_SIZE)
                    .permute(0, 2, 1)
                    .contiguous())                   # [8, 3072, 128]
            W_V_layers.append(W_V)
    return {"W_V": W_V_layers}


# ── Geometric health ───────────────────────────────────────────────────────────

def _stable_rank(W: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(W.float())
    return (sv ** 2).sum().item() / (sv[0] ** 2).item()

def _condition_number(W: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(W.float())
    return (sv[0] / sv[-1]).item()

def _isotropy(W: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(W.float())
    return (sv[-1] / sv[0]).item()

def _spectral_entropy(W: torch.Tensor) -> float:
    sv  = torch.linalg.svdvals(W.float())
    p   = sv ** 2 / (sv ** 2).sum()
    p   = p[p > 0]
    return (-(p * p.log())).sum().item()

def _effective_rank(W: torch.Tensor) -> float:
    return float(torch.exp(torch.tensor(_spectral_entropy(W))).item())


def _compute_geometric_health(adapted: dict) -> dict:
    W_V_layers = adapted["W_V"]   # List[28] of [8, 3072, 128]

    per_layer     = []
    all_sr        = []
    all_cond      = []
    all_iso       = []
    all_ent       = []
    all_er        = []

    for l, W_V_l in enumerate(W_V_layers):
        head_sr, head_cond, head_iso, head_ent, head_er = [], [], [], [], []

        for h in range(_N_KV_HEADS):
            W_h = W_V_l[h].float()   # [3072, 128]
            head_sr.append(_stable_rank(W_h))
            head_cond.append(_condition_number(W_h))
            head_iso.append(_isotropy(W_h))
            head_ent.append(_spectral_entropy(W_h))
            head_er.append(_effective_rank(W_h))

        layer_sr   = float(np.mean(head_sr))
        layer_cond = float(np.mean(head_cond))
        layer_iso  = float(np.mean(head_iso))
        layer_ent  = float(np.mean(head_ent))
        layer_er   = float(np.mean(head_er))

        all_sr.append(layer_sr)
        all_cond.append(layer_cond)
        all_iso.append(layer_iso)
        all_ent.append(layer_ent)
        all_er.append(layer_er)

        per_layer.append({
            "W_V": {
                "V_stable_rank":      layer_sr,
                "V_condition_number": layer_cond,
                "V_isotropy":         layer_iso,
                "V_sv_entropy":       layer_ent,
                "V_effective_rank":   layer_er,
            }
        })

    return {
        "per_layer": per_layer,
        "global": {
            "W_V": {
                "V_stable_rank":      float(np.mean(all_sr)),
                "V_condition_number": float(np.mean(all_cond)),
                "V_isotropy":         float(np.mean(all_iso)),
                "V_sv_entropy":       float(np.mean(all_ent)),
                "V_effective_rank":   float(np.mean(all_er)),
            }
        }
    }


# ── Recovery ───────────────────────────────────────────────────────────────────

def recover_run(run_dir: Path, method: str) -> bool:
    merged_geo = run_dir / "final" / "geometric_health_merged.pt"
    if merged_geo.exists():
        g = torch.load(merged_geo, map_location="cpu")
        sr = g["global"]["W_V"]["V_stable_rank"]
        log.info(f"  SKIP (already recovered)  sr={sr:.3f}")
        return True

    adapter_path = run_dir / "adapter_final.pt"
    if not adapter_path.exists():
        log.warning(f"  NO adapter_final.pt found: {run_dir}")
        return False

    rank = LORA_RANKS.get(method, 8)
    log.info(f"  Loading NF4 base + adapter (r={rank}) ...")

    try:
        from peft import LoraConfig, get_peft_model
        from paft.model.llama_paft_model import load_llama_nf4
        from peft import prepare_model_for_kbit_training

        base, _ = load_llama_nf4(DEFAULT_MODEL, device_map="auto")
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=False
        )

        lora_config = LoraConfig(
            r              = rank,
            lora_alpha     = rank * 2,
            lora_dropout   = 0.05,
            target_modules = ["v_proj", "o_proj"],
            bias           = "none",
            task_type      = "CAUSAL_LM",
        )
        model = get_peft_model(base, lora_config)

        # Load saved adapter weights
        state = torch.load(adapter_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            log.warning(f"  Unexpected keys: {unexpected[:3]}")
        log.info(f"  Loaded adapter — {len(state)} tensors")

        # Merge LoRA into base weights
        log.info(f"  Merging adapters ...")
        merged = model.merge_and_unload()
        merged.eval()

        # Extract per-head weights
        adapted    = _extract_llama_v_proj(merged)
        geo_health = _compute_geometric_health(adapted)

        # Save
        torch.save(adapted,    run_dir / "final" / "adapted_weights_merged.pt")
        torch.save(geo_health, merged_geo)

        sr = geo_health["global"]["W_V"]["V_stable_rank"]
        log.info(f"  OK  sr(W_V) = {sr:.3f}  → {merged_geo.name}")

        del merged, model, base
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        log.error(f"  FAILED: {e}", exc_info=True)
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global DEFAULT_MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/llama", type=Path)
    p.add_argument("--model_name",  default=DEFAULT_MODEL)
    args = p.parse_args()

    DEFAULT_MODEL = args.model_name

    ok = fail = 0
    for task in TASKS:
        task_dir = args.results_dir / task
        if not task_dir.exists():
            continue
        for method in LORA_METHODS:
            method_dir = task_dir / method
            if not method_dir.exists():
                continue
            sentinel = method_dir / "final" / "training_complete"
            if not sentinel.exists():
                log.warning(f"SKIP {task}/{method} — training_complete missing")
                continue

            log.info(f"Recovering {task}/{method}")
            if recover_run(method_dir, method):
                ok += 1
            else:
                fail += 1

    log.info(f"\nDone.  Recovered: {ok}  Failed: {fail}")

    # Print updated geometric health table
    if ok > 0:
        print("\nUpdated sr(W_V) after merge:")
        print(f"{'Method':<12}  {'Task':<14}  {'sr_final':>9}")
        print("─" * 42)
        for task in TASKS:
            for method in LORA_METHODS:
                p = args.results_dir / task / method / "final" / "geometric_health_merged.pt"
                if p.exists():
                    g  = torch.load(p, map_location="cpu")
                    sr = g["global"]["W_V"]["V_stable_rank"]
                    print(f"{method:<12}  {task:<14}  {sr:>9.3f}")


if __name__ == "__main__":
    main()

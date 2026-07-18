#!/usr/bin/env python3
"""
compute_lora_epoch_sr.py

Computes sr(W_eff) = sr(W_0 + scale * B @ A) for LoRA at each epoch
by loading HF checkpoint adapters and merging with base model weights.

Run from project root:
    python3 scripts/compute_lora_epoch_sr.py

Writes to:
    results/glue/{task}/{method}/epoch_{n}/geometric_health_merged.pt
"""

import json
import glob
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel
from safetensors.torch import load_file

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

BASE_MODEL   = "microsoft/deberta-v3-base"
RESULTS_DIR  = Path("results/glue")
TASKS        = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"]
METHODS      = ["lora_r8", "lora_r64"]
N_LAYERS     = 12
N_HEADS      = 12
D_HEAD       = 64   # d_h = d_model / n_heads = 768 / 12

# Map method name → LoRA rank and alpha
METHOD_CONFIG = {
    "lora_r8":  {"rank": 8,  "alpha": 16},
    "lora_r64": {"rank": 64, "alpha": 128},
}


# ─────────────────────────────────────────────────────────────
# Geometric health
# ─────────────────────────────────────────────────────────────

def geometric_health(W: torch.Tensor) -> dict:
    """Compute spectral metrics for a single 2D weight matrix."""
    W = W.float()
    sv = torch.linalg.svdvals(W)
    sr = (sv ** 2).sum().item() / (sv[0] ** 2).item()
    p  = sv ** 2 / (sv ** 2).sum()
    H  = -(p * torch.log(p + 1e-10)).sum().item()
    er   = float(np.exp(H))
    cond = (sv[0] / sv[-1]).item()
    iso  = (sv[-1] / sv[0]).item()
    return {
        "stable_rank":      sr,
        "spectral_entropy": H,
        "effective_rank":   er,
        "condition_number": cond,
        "isotropy":         iso,
    }


def analyze_full_matrix_per_head(W_full: torch.Tensor) -> tuple:
    """
    W_full: [768, 768] — full value_proj weight.
    Slice into N_HEADS heads of [D_HEAD, 768] and compute
    geometric health per head.
    Returns (per_head list of dicts, mean_sr float).
    """
    assert W_full.dim() == 2, f"Expected 2D, got {W_full.shape}"
    # Each head slice: rows [h*D_HEAD : (h+1)*D_HEAD]
    per_head = []
    srs = []
    for h in range(N_HEADS):
        W_head = W_full[h * D_HEAD : (h + 1) * D_HEAD, :]  # [64, 768]
        gh = geometric_health(W_head)
        per_head.append(gh)
        srs.append(gh["stable_rank"])
    return per_head, float(np.mean(srs))


# ─────────────────────────────────────────────────────────────
# Checkpoint → epoch mapping
# ─────────────────────────────────────────────────────────────

def get_checkpoint_epoch_map(method_dir: Path) -> dict:
    """
    Returns {epoch_int: Path_to_adapter_safetensors}.
    Reads trainer_state.json from each HF checkpoint to find epoch.
    """
    mapping = {}
    for state_file in sorted(
        method_dir.glob("hf_checkpoints/*/trainer_state.json")
    ):
        try:
            with open(state_file) as f:
                state = json.load(f)
            epoch = int(round(float(state["epoch"])))
            adapter_file = state_file.parent / "adapter_model.safetensors"
            if adapter_file.exists():
                mapping[epoch] = adapter_file
        except Exception as e:
            print(f"  Warning: could not read {state_file}: {e}")
    return mapping


# ─────────────────────────────────────────────────────────────
# Merge and compute
# ─────────────────────────────────────────────────────────────

def compute_merged_sr(
    base_state: dict,
    adapter_path: Path,
    rank: int,
    alpha: float,
) -> dict:
    """
    Load LoRA adapter from safetensors, merge with base weights,
    compute geometric health per layer per head.
    Returns result dict.
    """
    scale = alpha / rank
    adapter = load_file(adapter_path)

    per_layer_V = []
    all_sr_V    = []

    for layer_idx in range(N_LAYERS):
        # Base weight key
        base_key = (
            f"encoder.layer.{layer_idx}"
            f".attention.self.value_proj.weight"
        )
        # LoRA keys in HF adapter
        lora_a_key = (
            f"base_model.model.deberta.encoder.layer.{layer_idx}"
            f".attention.self.value_proj.lora_A.weight"
        )
        lora_b_key = (
            f"base_model.model.deberta.encoder.layer.{layer_idx}"
            f".attention.self.value_proj.lora_B.weight"
        )

        W0 = base_state.get(base_key)
        if W0 is None:
            print(f"  Warning: base key not found: {base_key}")
            continue

        W0 = W0.float()

        if lora_a_key in adapter and lora_b_key in adapter:
            A = adapter[lora_a_key].float()  # [rank, 768]
            B = adapter[lora_b_key].float()  # [768, rank]
            W_eff = W0 + scale * (B @ A)    # [768, 768]
        else:
            # Do NOT silently fall back to the base weight — that computes
            # sr(W_0), not sr(W_eff), and writes it into per-epoch data
            # indistinguishably from a real merged-weight measurement. This
            # is what produced the flat-at-pretrained-value bug that
            # patch_metrics_cache.py had to detect and patch after the fact.
            # Skip the layer instead; it will be missing from per_layer_V,
            # and callers must treat that as missing data, not zero shift.
            print(
                f"  Warning: LoRA keys not found for layer {layer_idx} — "
                f"skipping this layer (NOT substituting base weight)"
            )
            continue

        per_head, mean_sr = analyze_full_matrix_per_head(W_eff)
        per_layer_V.append({
            "layer":      layer_idx,
            "mean_sr":    mean_sr,
            "per_head":   per_head,
        })
        all_sr_V.append(mean_sr)

    n_total = N_LAYERS
    n_ok    = len(all_sr_V)
    if n_ok < n_total:
        print(f"  Warning: only {n_ok}/{n_total} layers had usable LoRA adapter "
              f"weights for this checkpoint — result is a partial average, "
              f"not all-layer coverage")
    mean_sr_V = float(np.mean(all_sr_V)) if all_sr_V else None

    return {
        "mean_sr_V":    mean_sr_V,
        "mean_sr_O":    None,   # LoRA on DeBERTa targets query+value, not out_proj
        "mean_sr":      mean_sr_V,
        "per_layer_V":  per_layer_V,
        "per_layer_O":  [],
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    from analysis.utils import setup_run_log
    setup_run_log("compute_lora_epoch_sr")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute and overwrite geometric_health_merged.pt even if it "
             "already exists. Use this after any fix to the merge/skip logic "
             "in this script — existing output files silently short-circuit "
             "recomputation otherwise, so a code fix has no effect on disk "
             "until stale files are regenerated.",
    )
    args = parser.parse_args()
    print(f"Mode: {'FORCE (overwriting existing files)' if args.force else 'normal (skip existing files)'}")

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModel.from_pretrained(BASE_MODEL)
    base_state = base_model.state_dict()
    print(f"Base model loaded. {len(base_state)} keys.\n")

    total_written  = 0
    total_skipped  = 0
    total_missing  = 0

    for task in TASKS:
        for method in METHODS:
            method_dir = RESULTS_DIR / task / method
            if not method_dir.exists():
                print(f"Skipping {task}/{method} — directory not found")
                continue

            cfg   = METHOD_CONFIG[method]
            rank  = cfg["rank"]
            alpha = cfg["alpha"]

            # Get epoch → checkpoint mapping from HF checkpoints
            epoch_map = get_checkpoint_epoch_map(method_dir)
            if not epoch_map:
                print(f"{task}/{method}: no HF checkpoints found")
                total_missing += 1
                continue

            print(f"{task}/{method} "
                  f"(rank={rank}, alpha={alpha}, "
                  f"scale={alpha/rank:.1f}, "
                  f"epochs={sorted(epoch_map.keys())})")

            for epoch, adapter_path in sorted(epoch_map.items()):
                epoch_dir = method_dir / f"epoch_{epoch}"
                epoch_dir.mkdir(exist_ok=True)

                out_path = epoch_dir / "geometric_health_merged.pt"

                if out_path.exists() and not args.force:
                    print(f"  Epoch {epoch}: already exists — skipping (use --force to recompute)")
                    total_skipped += 1
                    continue

                print(
                    f"  Epoch {epoch}: merging from "
                    f"{adapter_path.parent.name}...",
                    end=" ", flush=True
                )

                result = compute_merged_sr(
                    base_state, adapter_path, rank, alpha
                )

                torch.save(result, out_path)
                total_written += 1

                sr_display = f"{result['mean_sr_V']:.3f}" if result['mean_sr_V'] is not None else "None (all layers skipped)"
                print(f"sr(W_V)={sr_display}")

    print(
        f"\nDone. "
        f"Written: {total_written}  "
        f"Skipped: {total_skipped}  "
        f"Missing: {total_missing}"
    )
    print("Run generate_paper_outputs.py to rebuild tables.")


if __name__ == "__main__":
    main()
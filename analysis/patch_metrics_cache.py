#!/usr/bin/env python3
"""
patch_metrics_cache.py

Patches metrics_cache.json with correct per-epoch sr(W_eff) values
for LoRA and PoLAR, read from geometric_health_merged.pt files.

Run from project root:
    python3 scripts/patch_metrics_cache.py
"""

import json
import torch
import numpy as np
from pathlib import Path

RESULTS_DIR  = Path("results/glue")
CACHE_PATH   = Path("results/analysis/metrics_cache.json")
TASKS        = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"]
METHODS      = ["lora_r8", "lora_r64", "polar_r8"]


def get_epoch_dirs(method_dir: Path) -> dict:
    """Returns {epoch_int: Path} for all epoch_N directories."""
    result = {}
    for ep_dir in sorted(method_dir.glob("epoch_*")):
        try:
            ep = int(ep_dir.name.split("_")[1])
            result[ep] = ep_dir
        except (ValueError, IndexError):
            continue
    return result


def load_merged_sr(epoch_dir: Path) -> float | None:
    """Load mean_sr_V from geometric_health_merged.pt."""
    merged_path = epoch_dir / "geometric_health_merged.pt"
    if not merged_path.exists():
        return None
    data = torch.load(merged_path, map_location="cpu", weights_only=True)
    return data.get("mean_sr_V")


def main():
    print(f"Loading cache: {CACHE_PATH}")
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    glue = cache["glue"]
    patched = 0
    skipped = 0

    for task in TASKS:
        for method in METHODS:
            method_dir = RESULTS_DIR / task / method
            if not method_dir.exists():
                continue

            entry = glue.get(task, {}).get(method)
            if entry is None:
                print(f"  {task}/{method}: not in cache — skipping")
                skipped += 1
                continue

            epoch_dirs = get_epoch_dirs(method_dir)
            if not epoch_dirs:
                print(f"  {task}/{method}: no epoch dirs found")
                skipped += 1
                continue

            # Build corrected per_epoch list
            per_epoch_corrected = []
            for ep in sorted(epoch_dirs.keys()):
                sr = load_merged_sr(epoch_dirs[ep])
                if sr is not None:
                    per_epoch_corrected.append({
                        "epoch": ep,
                        "sr_Weff": sr,
                    })

            if not per_epoch_corrected:
                print(f"  {task}/{method}: no merged files found")
                skipped += 1
                continue

            # Check if old values were wrong (flat at pretrained 34.745)
            old_per_epoch = entry.get("per_epoch", [])
            had_wrong = False
            if isinstance(old_per_epoch, list) and old_per_epoch:
                had_wrong = any(
                    abs(v.get("sr_Weff", 0) - 34.745) < 0.01
                    for v in old_per_epoch
                )

            # Patch per_epoch
            entry["per_epoch"] = per_epoch_corrected

            # Patch sr_Weff_final with last epoch value
            last_sr = per_epoch_corrected[-1]["sr_Weff"]
            old_final = entry.get("sr_Weff_final")
            entry["sr_Weff_final"] = last_sr

            status = "FIXED" if had_wrong else "updated"
            old_str = f"{old_final:.3f}" if old_final is not None else "None"
            print(
                f"  {task}/{method}: {status} — "
                f"{len(per_epoch_corrected)} epochs, "
                f"final sr={last_sr:.3f} (was {old_str})"
            )
            patched += 1

    # Write back
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\nDone. Patched: {patched}  Skipped: {skipped}")
    print(f"Cache updated: {CACHE_PATH}")
    print("Run generate_paper_outputs.py to rebuild all tables.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
discover_epoch_files.py

Before plotting lam/delta_sigma trajectories, we need to know what's
actually saved per epoch. This lists every file in a sample epoch_N
directory for a PAFT method and SVF, on whichever tasks/models you have,
and reports whether the raw parameters (lam, S) or just derived metrics
(geometric_health.pt) are available.

Usage:
    python3 discover_epoch_files.py
"""
from pathlib import Path

import torch

SAMPLE_DEBERTA_TASK = "sst2"   # has 5 epochs, good trajectory length
SAMPLE_LLAMA_TASK = "boolq"

CHECKS = [
    ("DeBERTa pure_paft", Path(f"results/glue/{SAMPLE_DEBERTA_TASK}/pure_paft")),
    ("DeBERTa hybrid_paft", Path(f"results/glue/{SAMPLE_DEBERTA_TASK}/hybrid_paft")),
    ("DeBERTa svf", Path(f"results/glue/{SAMPLE_DEBERTA_TASK}/svf")),
    ("LLaMA pure_paft", Path(f"results/llama/{SAMPLE_LLAMA_TASK}/pure_paft")),
    ("LLaMA hybrid_paft", Path(f"results/llama/{SAMPLE_LLAMA_TASK}/hybrid_paft")),
]


def inspect_epoch_dir(method_dir: Path):
    epoch_dirs = sorted(
        method_dir.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1,
    )
    if not epoch_dirs:
        print("  no epoch_* directories found")
        return

    print(f"  {len(epoch_dirs)} epoch dirs: {[d.name for d in epoch_dirs]}")
    sample = epoch_dirs[0]
    print(f"  contents of {sample}:")
    for f in sorted(sample.iterdir()):
        print(f"    {f.name}")
        if f.suffix == ".pt":
            try:
                data = torch.load(f, map_location="cpu", weights_only=True)
                if isinstance(data, dict):
                    keys = list(data.keys())
                    print(f"      keys: {keys[:15]}{' ...' if len(keys) > 15 else ''}")
                    for k in ("lam_V", "lam_O", "S_V", "S_O", "Q_V", "Q_O"):
                        if k in data:
                            v = data[k]
                            shape = v[0].shape if isinstance(v, list) else getattr(v, "shape", "?")
                            print(f"        FOUND raw param '{k}': shape {shape}")
            except Exception as e:
                print(f"      (could not load: {e})")


def main():
    for label, method_dir in CHECKS:
        print(f"\n{'='*70}\n{label}: {method_dir}\n{'='*70}")
        if not method_dir.exists():
            print("  directory does not exist, skipping")
            continue
        inspect_epoch_dir(method_dir)


if __name__ == "__main__":
    main()
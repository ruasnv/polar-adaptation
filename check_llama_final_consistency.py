#!/usr/bin/env python3
"""
check_llama_final_consistency.py

After manually deleting extra epoch directories (e.g. removing a stray
epoch_2), 'final/' may still hold results from the ORIGINAL run (the one
with the wrong epoch count), while the highest surviving epoch_N/ holds
results from what you actually want to report. This script flags every
task/method where final/metrics.json and the latest surviving epoch's
metrics.json disagree, so you know exactly what to fix before touching
any table.

This is a read-only report. It does not modify anything.

Usage:
    python3 check_llama_final_consistency.py --results_dir results/llama
"""
import argparse
import json
from pathlib import Path

TASKS = ["boolq", "hellaswag", "arc_challenge"]
METHODS = ["frozen", "pure_paft", "hybrid_paft", "lora_r8", "polar_r8",
           "lora_r64", "bitfit", "safe_pure_paft", "safe_hybrid_paft", "svf", "full_ft"]


def load_metric(path: Path):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        for key in ("final_accuracy", "accuracy"):
            if key in d:
                return d[key]
    except Exception as e:
        print(f"    (error reading {path}: {e})")
    return None


def latest_epoch_dir(method_dir: Path):
    epoch_dirs = sorted(
        method_dir.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1,
    )
    return epoch_dirs[-1] if epoch_dirs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/llama", type=Path)
    args = p.parse_args()

    mismatches = []
    checked = 0

    for task in TASKS:
        for method in METHODS:
            method_dir = args.results_dir / task / method
            if not method_dir.exists():
                continue

            final_p = method_dir / "final" / "metrics.json"
            final_acc = load_metric(final_p)

            latest_dir = latest_epoch_dir(method_dir)
            latest_acc = None
            latest_name = None
            if latest_dir is not None:
                latest_name = latest_dir.name
                latest_acc = load_metric(latest_dir / "metrics.json")

            if final_acc is None and latest_acc is None:
                continue
            checked += 1

            print(f"{task}/{method}")
            print(f"  final/metrics.json:            {final_acc}")
            print(f"  {latest_name or '(no epoch dirs)'}/metrics.json:"
                  f"{'':<{max(1, 20-len(str(latest_name)))}}{latest_acc}")

            if final_acc is not None and latest_acc is not None:
                if abs(final_acc - latest_acc) > 1e-6:
                    print(f"  *** MISMATCH — final and latest epoch disagree ***")
                    mismatches.append((task, method, final_acc, latest_acc, latest_name))
                else:
                    print(f"  OK — consistent")
            print()

    print("=" * 70)
    print(f"Checked {checked} task/method combinations. "
          f"Found {len(mismatches)} mismatch(es).")
    if mismatches:
        print("\nMismatched entries (task, method, final_acc, latest_epoch_acc, latest_epoch_name):")
        for m in mismatches:
            print(f"  {m}")
        print("\nFor each mismatch: decide which is the checkpoint you actually "
              "want (almost certainly the latest surviving epoch, since 'final' "
              "may be a leftover from the original wrong-epoch-count run), then "
              "either re-point 'final' at it or update collect_llama_results.sh "
              "/ recover_lora_weights_llama.py to read from the correct "
              "directory before regenerating any LLaMA table.")


if __name__ == "__main__":
    main()
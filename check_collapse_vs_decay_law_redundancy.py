#!/usr/bin/env python3
"""
check_collapse_vs_decay_law_redundancy.py

Confirms numerically whether collapse.pdf and decay_law.pdf plot the same
underlying data. collapse.pdf reads sr_Weff_final directly from
metrics_cache.json per task/method. decay_law.pdf's scatter points come
from decay_law_results.json's "steps"/"sr_values" arrays, built by
fit_decay_law.py from the SAME metrics_cache.json field. This checks that
those two sources actually agree exactly for every method/task they share,
rather than assuming it from reading the code.

Usage:
    python3 check_collapse_vs_decay_law_redundancy.py
"""
import json
from pathlib import Path

CACHE_PATH = Path("results/analysis/metrics_cache.json")
DECAY_PATH = Path("results/analysis/decay_law_results.json")

STEPS = {
    "rte":  780, "mrpc": 1150, "stsb": 1800, "cola": 2670,
    "sst2": 10520, "qnli": 16365, "qqp": 34110, "mnli": 36816,
}


def main():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    glue = cache.get("glue", {})

    with open(DECAY_PATH) as f:
        decay = json.load(f)
    decay_methods = decay.get("methods", {})

    step_to_task = {v: k for k, v in STEPS.items()}

    total_compared = 0
    total_mismatch = 0
    methods_checked = []

    for method, fit in decay_methods.items():
        if fit is None:
            continue
        methods_checked.append(method)
        decay_steps = fit.get("steps", [])
        decay_sr = fit.get("sr_values", [])

        print(f"\n── {method} ──")
        for step, sr_from_decay in zip(decay_steps, decay_sr):
            task = step_to_task.get(step)
            if task is None:
                print(f"  step={step}: no matching task found, skipping")
                continue

            entry = glue.get(task, {}).get(method, {})
            sr_from_cache = entry.get("sr_Weff_final")

            total_compared += 1
            if sr_from_cache is None:
                print(f"  {task}: cache has no sr_Weff_final for this method — MISMATCH (missing)")
                total_mismatch += 1
                continue

            match = abs(sr_from_cache - sr_from_decay) < 1e-9
            status = "MATCH" if match else "MISMATCH"
            if not match:
                total_mismatch += 1
            print(f"  {task}: collapse.pdf source={sr_from_cache}  "
                  f"decay_law.pdf source={sr_from_decay}  [{status}]")

    print(f"\n{'='*60}")
    print(f"Methods checked: {methods_checked}")
    print(f"Total task/method points compared: {total_compared}")
    print(f"Mismatches: {total_mismatch}")
    if total_mismatch == 0 and total_compared > 0:
        print("\nVERDICT: collapse.pdf and decay_law.pdf's scatter points are "
              "numerically IDENTICAL for every method/task compared. "
              "collapse.pdf's data is a strict subset of decay_law.pdf's — "
              "same points, minus the fitted curve and decay-rate summary. "
              "Confirmed redundant, not just visually similar.")
    elif total_compared == 0:
        print("\nVERDICT: no comparable points found — check that both "
              "files exist and were generated from the same cache.")
    else:
        print("\nVERDICT: real mismatches found — NOT simply redundant. "
              "Investigate before dropping either plot.")


if __name__ == "__main__":
    main()
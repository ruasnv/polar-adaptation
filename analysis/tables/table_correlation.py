#!/usr/bin/env python3
"""
analysis/table_correlation.py

Computes task-isolated statistical correlations between final geometric
health indicators and downstream performance scores.
"""
import json
from pathlib import Path
from scipy.stats import pearsonr
import numpy as np


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        print("Error: Compile metrics_cache.json before running this script.")
        return

    with open(cache_path) as f:
        data = json.load(f)["glue"]

    print("\nTable: Task-Isolated Geometric Performance Correlations (Pearson r)")
    print("─" * 70)
    print(f"{'GLUE Task':<12} | {'sr(W_eff) Correlation r':<24} | {'p-value':<8}")
    print("─" * 70)

    global_r_samples = []

    for task, methods in data.items():
        task_scores = []
        task_sr_weff = []

        for method, metrics in methods.items():
            if "frozen" in method: continue
            task_scores.append(metrics["task_score"])
            task_sr_weff.append(metrics["sr_Weff_final"])

        if len(task_scores) < 3: continue

        try:
            r_val, p_val = pearsonr(task_sr_weff, task_scores)
            print(f"{task:<12} | {r_val:<24.4f} | {p_val:<8.2e}")
            if not np.isnan(r_val):
                global_r_samples.append(r_val)
        except Exception:
            print(f"{task:<12} | Correlation unstable   | N/A")

    print("─" * 70)
    if global_r_samples:
        print(f"Mean Task-Isolated Pearson r: {np.mean(global_r_samples):.4f}")
    print("─" * 70)


if __name__ == "__main__":
    main()
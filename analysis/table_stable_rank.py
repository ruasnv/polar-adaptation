#!/usr/bin/env python3
"""
analysis/table_stable_rank.py

Aggregates and prints mean Stable Rank metrics across tasks to format Table 2.
"""
import json
import os
from pathlib import Path
import numpy as np


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        print("Error: Compile metrics_cache.json before running this script.")
        return

    with open(cache_path) as f:
        data = json.load(f)["glue"]

    methods_summary = {}

    for task, methods in data.items():
        for method, metrics in methods.items():
            if method not in methods_summary:
                methods_summary[method] = {"init": [], "final": [], "delta": []}

            init_sr = metrics["sr_Weff_init"]
            final_sr = metrics["sr_Weff_final"]

            methods_summary[method]["init"].append(init_sr)
            methods_summary[method]["final"].append(final_sr)
            methods_summary[method]["delta"].append(final_sr - init_sr)

    print("\nTable 2: Maintained and Evolving Stable Rank Metrics across GLUE Matrix")
    print("─" * 75)
    print(f"{'Fine-Tuning Method':<28} | {'sr(W_eff) Init':<14} | {'sr(W_eff) Final':<15} | {'Mean Δsr':<9}")
    print("─" * 75)

    latex_lines = []

    # Order methods alphabetically or logically by category for structural presentation
    for method in sorted(methods_summary.keys()):
        m_init = np.mean(methods_summary[method]["init"])
        m_final = np.mean(methods_summary[method]["final"])
        m_delta = np.mean(methods_summary[method]["delta"])

        print(f"{method:<28} | {m_init:<14.3f} | {m_final:<15.3f} | {m_delta:<+9.3f}")

        # Format strings clean for raw LaTeX compilation
        tex_method = method.replace("_", r"\_")
        latex_lines.append(f"{tex_method} & {m_init:.3f} & {m_final:.3f} & {m_delta:+.3f} \\\\")

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table_stable_rank.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print("─" * 75)
    print("LaTeX document formatting segments successfully compiled to results/analysis/table_stable_rank.tex")


if __name__ == "__main__":
    main()
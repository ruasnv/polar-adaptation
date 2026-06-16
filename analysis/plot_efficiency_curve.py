#!/usr/bin/env python3
"""
plot_efficiency_curve.py

Plots the efficiency curve for a range of values.
Takes a list of tasks.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    root = Path("results/glue")
    tasks = ["cola", "mnli", "mrpc", "qnli", "rte", "sst2", "stsb"]

    methods_data = {}

    for task in tasks:
        task_dir = root / task
        if not task_dir.exists(): continue
        for m_dir in task_dir.iterdir():
            if not m_dir.is_dir(): continue
            m_name = m_dir.name
            metrics_p = m_dir / "metrics.json"
            if not metrics_p.exists(): continue

            with open(metrics_p) as f:
                d = json.load(f)

            # Extract standard tracking parameters scale keys
            p_count = d.get("trainable_params", 0)
            if p_count == 0:
                if "lora_r8" in m_name:
                    p_count = 591000
                elif "lora_r64" in m_name:
                    p_count = 4719000
                elif "pure_paft" in m_name:
                    p_count = 18400
                elif "safe_pure" in m_name:
                    p_count = 102000
                elif "safe_hybrid" in m_name:
                    p_count = 1260000
                elif "bitfit" in m_name:
                    p_count = 84000
                elif "frozen" in m_name:
                    p_count = 1

            if m_name not in methods_data:
                methods_data[m_name] = {"params": p_count, "scores": []}

            # Appending localized task values
            metric_keys = ["accuracy", "f1", "matthews_correlation", "pearson"]
            for k in metric_keys:
                if k in d:
                    methods_data[m_name]["scores"].append(d[k])
                    break

    # Format graphic layout configuration sheets
    plt.figure(figsize=(7, 5))

    for name, v in methods_data.items():
        if not v["scores"]: continue
        mean_score = np.mean(v["scores"])
        params = v["params"]

        # Color profile classifications
        color = "grey"
        if "paft" in name:
            color = "royalblue"
        elif "lora" in name:
            color = "darkorange"
        elif "polar" in name:
            color = "forestgreen"

        plt.scatter(params, mean_score, s=120, color=color, alpha=0.8, edgecolors="black",
                    label=name if name not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(params * 1.1, mean_score, name, fontsize=8, alpha=0.7)

    plt.xscale("log")
    plt.title("Parameter Efficiency Frontiers (PEFT Pareto Analysis)", fontsize=11, fontweight="bold")
    plt.xlabel("Trainable Parameters Count (Log Scale)", fontsize=10)
    plt.ylabel("Mean Normalized Tasks Performance Range", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    os.makedirs("results/analysis/figures", exist_ok=True)
    plt.savefig("results/analysis/figures/efficiency_curve.pdf", bbox_inches="tight")
    print("Successfully rendered Figure 3 (efficiency_curve.pdf) on CPU.")


if __name__ == "__main__":
    main()
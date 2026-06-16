#!/usr/bin/env python3
"""
analysis/plot_layer_profiles_delta.py

Plots per-layer evolution of transformation magnitudes over epochs across tasks.
"""
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def setup_iclr_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['font.size'] = 10


def main():
    setup_iclr_style()
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        print("Error: Compile metrics_cache.json before running this script.")
        return

    with open(cache_path) as f:
        data = json.load(f)["glue"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)

    target_methods = {
        "safe_hybrid_paft": ("#810f7c", "-",  "o", "safe-hybrid-PAFT"),
        "hybrid_paft":      ("#4a90e2", "--", "s", "hybrid-PAFT"),
        "polar_r8":         ("#2ca02c", "-.", "^", "PoLAR r=8"),
        "full_ft":          ("#d62728", ":",  "x", "Full FT"),
    }

    for task, ax in [("cola", ax1), ("mrpc", ax2)]:
        if task not in data:
            continue
        # sr_Weff_init is identical across methods (same pretrained weights)
        # read it from any method that has it
        init_sr = next(
            v["sr_Weff_init"] for v in data[task].values()
            if "sr_Weff_init" in v
        )
        for method_name, (color, ls, marker, label) in target_methods.items():
            if method_name not in data[task]:
                continue
            layers = data[task][method_name]["per_layer"]
            if not layers:
                continue
            x = [l["layer"] for l in layers]
            y = [l["sr_Weff_final"] - init_sr for l in layers]
            # Only add legend labels on the first panel to avoid duplicates
            ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                    markersize=3, linewidth=1.2,
                    label=label if task == "cola" else "")

    ax1.set_title("CoLA (Syntactic)", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Layer Index", fontsize=9)
    ax1.set_ylabel(r"$\Delta sr(W_{\mathrm{eff}})$ vs. Pretrained", fontsize=9)
    ax1.axhline(0, color='black', linewidth=0.6, linestyle='-')
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.set_xticks(range(12))

    ax2.set_title("MRPC (Semantic)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Layer Index", fontsize=9)
    ax2.axhline(0, color='black', linewidth=0.6, linestyle='-')
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.set_xticks(range(12))

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10),
               ncol=4, fontsize=8.5, frameon=True, edgecolor='#e0e0e0')

    plt.tight_layout()
    out_figures = Path("results/analysis/figures")
    out_figures.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_figures / "layer_profiles_delta.pdf", bbox_inches="tight")
    print("Successfully rendered layer_profiles_delta.pdf")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
analysis/plot_layer_profile.py

Generates Figure 2 for the main text. Compiles a two-panel panel tracking
layer-wise geometric deltas across syntax (CoLA) and semantic (MRPC) fields.
"""
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path


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

    target_methods = ["safe_hybrid_paft", "hybrid_paft", "polar_r8", "svf", "full_ft"]
    method_styles = {
        "safe_hybrid_paft": ("#002f6c", "-", "o"),
        "hybrid_paft": ("#4a90e2", "--", "s"),
        "polar_r8": ("#2ca02c", ":", "d"),
        "svf": ("#9467bd", "-.", "^"),
        "full_ft": ("#d62728", "-", "x")
    }
    method_labels = {
        "safe_hybrid_paft": "safe-hybrid-PAFT (Ours)", "hybrid_paft": "hybrid-PAFT",
        "polar_r8": "PoLAR (r=8)", "svf": "SVF Baseline", "full_ft": "Full Fine-Tuning"
    }

    # Panel 1 execution: CoLA
    if "cola" in data:
        for m in target_methods:
            if m not in data["cola"]: continue
            # Replace both panel loops with this pattern:
            layers = data["cola"][m]["per_layer"]
            x, y = [], []
            for l in layers:
                final_sr = l.get("sr_Weff_final")
                if final_sr is None:
                    continue
                x.append(l["layer"])
                y.append(final_sr - data["cola"][m]["sr_Weff_init"])
            if not x:
                continue  # skip this method entirely if no valid layers
            ax1.plot(x, y, ...)
            color, ls, marker = method_styles[m]
            ax1.plot(x, y, color=color, linestyle=ls, marker=marker, markersize=3.5, linewidth=1.1,
                     label=method_labels[m])

    ax1.set_title("CoLA (Syntactic Domain)", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Layer Index (Encoder Depth)", fontsize=9)
    ax1.set_ylabel(r"Geometric Delta ($\Delta$ Stable Rank $W_{\mathrm{eff}}$)", fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax1.set_xticks(range(12))

    # Panel 2 execution: MRPC
    if "mrpc" in data:
        for m in target_methods:
            if m not in data["mrpc"]: continue
            # Replace both panel loops with this pattern:
            layers = data["mrpc"][m]["per_layer"]
            x, y = [], []
            for l in layers:
                final_sr = l.get("sr_Weff_final")
                if final_sr is None:
                    continue
                x.append(l["layer"])
                y.append(final_sr - data["mrpc"][m]["sr_Weff_init"])
            if not x:
                continue  # skip this method entirely if no valid layers
            ax1.plot(x, y, color=color, linestyle=ls, marker=marker,
                     markersize=3.5, linewidth=1.2)
            color, ls, marker = method_styles[m]
            ax2.plot(x, y, color=color, linestyle=ls, marker=marker, markersize=3.5, linewidth=1.1)

    ax2.set_title("MRPC (Semantic Domain)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Layer Index (Encoder Depth)", fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax2.set_xticks(range(12))

    # Single consolidated horizontal legend placed cleanly above panels
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=8.5, frameon=True, edgecolor='#e0e0e0')

    plt.tight_layout()
    out_figures = Path("results/analysis/figures")
    out_figures.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_figures / "layer_profile.pdf", bbox_inches="tight")
    print("Successfully rendered Figure 2: layer_profile.pdf (ICLR Dual-Panel Standard)")


if __name__ == "__main__":
    main()
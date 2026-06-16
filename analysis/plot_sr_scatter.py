#!/usr/bin/env python3
"""
analysis/plot_sr_scatter.py

Generates Figure 1 for the main text using a broken y-axis layout. Maps
update complexity sr(Delta W) against effective weight geometry sr(W_eff).
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

    # Build a broken y-axis figure: top panel for LoRA, bottom panel for PAFT/Base
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(6.2, 4.5),
                                         gridspec_kw={'height_ratios': [1, 2.5]})
    fig.subplots_adjust(hspace=0.12)  # Close the gap between panels

    PRETRAINED_BASE = 34.745
    legend_tracker = {}

    family_colors = {
        "pure_paft": "#1f77b4", "safe_pure_paft": "#0c4da2",
        "hybrid_paft": "#4a90e2", "safe_hybrid_paft": "#002f6c",
        "lora_r8": "#ff7f0e", "lora_r64": "#b15900",
        "polar_r8": "#2ca02c", "svf": "#9467bd", "bitfit": "#7f7f7f"
    }

    family_labels = {
        "pure_paft": "pure-PAFT (Ours)", "safe_pure_paft": "safe-pure-PAFT (Ours)",
        "hybrid_paft": "hybrid-PAFT (Ours)", "safe_hybrid_paft": "safe-hybrid-PAFT (Ours)",
        "lora_r8": "LoRA (r=8)", "lora_r64": "LoRA (r=64)",
        "polar_r8": "PoLAR (r=8)", "svf": "SVF Baseline", "bitfit": "BitFit"
    }

    for task, methods in data.items():
        for method, metrics in methods.items():
            if method not in family_colors: continue

            x = metrics["sr_deltaW_V"]
            y = metrics["sr_Weff_final"]
            if "bitfit" in method: x = 0.0

            color = family_colors[method]
            label = family_labels[method] if method not in legend_tracker else ""
            legend_tracker[method] = True

            # Plot the identical points on both subplots; clipping bounds handle visibility
            ax_top.scatter(x, y, color=color, s=55, alpha=0.85, edgecolors='black', linewidths=0.5, label=label,
                           zorder=4)
            ax_bot.scatter(x, y, color=color, s=55, alpha=0.85, edgecolors='black', linewidths=0.5, zorder=4)

    # Set explicit coordinate limits for the broken segments
    ax_top.set_ylim(280, 360)  # Upper panel captures LoRA explosion range
    ax_bot.set_ylim(15, 45)  # Lower panel captures PAFT, PoLAR, and Base points

    # Draw horizontal target line across the lower pane
    ax_bot.axhline(y=PRETRAINED_BASE, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.8,
                   label="Pretrained Baseline", zorder=2)

    # Hide the interior bounding spines between panels
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)  # Don't draw tick labels on top edge
    ax_bot.xaxis.tick_bottom()

    # Draw the diagnostic cut-out hash lines on the bounding spines
    d = .012  # hash mark scale factor
    kwargs = dict(transform=ax_top.transAxes, color='black', clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)  # Top left spine break mark
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top right spine break mark

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom left spine break mark
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom right spine break mark

    # Assign labels and design details
    ax_bot.set_xlabel(r"Stable Rank of Parameter Update $\Delta W_V$ (Structural Complexity)", fontsize=10)
    fig.text(0.02, 0.5, r"Stable Rank of Effective Weight $W_{\mathrm{eff}}$ (Geometric Health)",
             va='center', rotation='vertical', fontsize=10)

    ax_top.set_title("Geometric Landscape: Update Space vs. Effective Weight Space", fontsize=11, fontweight='bold',
                     pad=14)
    ax_top.grid(True, linestyle=':', alpha=0.3)
    ax_bot.grid(True, linestyle=':', alpha=0.3)

    # Collect and compile clean joint legend block across axes lines
    handles, labels = ax_top.get_legend_handles_labels()
    h_b, l_b = ax_bot.get_legend_handles_labels()
    handles.extend(h_b)
    labels.extend(l_b)

    ax_bot.legend(handles, labels, loc="lower right", fontsize=8, frameon=True, facecolor='white', edgecolor='#e0e0e0')

    out_figures = Path("results/analysis/figures")
    out_figures.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_figures / "sr_scatter.pdf", bbox_inches="tight", dpi=300)
    print("Successfully rendered Figure 1: sr_scatter.pdf (ICLR Broken-Axis Standard)")


if __name__ == "__main__":
    main()
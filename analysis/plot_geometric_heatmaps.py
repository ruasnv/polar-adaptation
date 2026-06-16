#!/usr/bin/env python3
"""
analysis/plot_geometric_heatmaps.py

Generates dense structural evaluation heatmaps mapping layer indexes vs methods,
color-coded by relative percentage changes from the pretrained foundation.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def setup_iclr_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Computer Modern Serif', 'DejaVu Serif']
    plt.rcParams['font.size'] = 9


def main():
    setup_iclr_style()
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        print("Error: Compile metrics_cache.json before running this script.")
        return

    with open(cache_path) as f:
        data = json.load(f)["glue"]

    task = "stsb" if "stsb" in data else list(data.keys())[0]
    methods = sorted(list(data[task].keys()))

    # Pack metrics matrix array
    heatmap_matrix = np.zeros((len(methods), 12))

    for m_idx, method in enumerate(methods):
        m_data = data[task][method]
        init_sr = m_data["sr_Weff_init"]
        layers = m_data["per_layer"]

        for l_idx, layer in enumerate(layers):
            if l_idx >= 12: break
            final_sr = layer["sr_Weff_final"]
            # Percent change formulation
            heatmap_matrix[m_idx, l_idx] = ((final_sr - init_sr) / init_sr) * 100

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Using a professional diverging palette: Red = Altered/Exploded, Blue = Preserved
    cax = ax.imshow(heatmap_matrix, cmap="RdBu_r", aspect="auto", vmin=-50, vmax=50)

    # Custom colorbar integration
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(r"Relative Stable Rank Deviation from Pretrained State $\Delta sr$ (%)", fontsize=9)

    ax.set_xticks(range(12))
    ax.set_xlabel("Layer Index (Encoder Stack Depth)", fontsize=9.5)

    # Formatting display names clean for text tracking
    display_labels = [m.replace("_", "-") for m in methods]
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(display_labels, fontsize=8)

    ax.set_title(f"Global Geometric Health Map (Layer-wise Transformation Deviances)", fontsize=11, fontweight='bold',
                 pad=12)

    plt.tight_layout()
    out_figures = Path("results/analysis/figures")
    out_figures.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_figures / "geometric_heatmaps.pdf", bbox_inches="tight", dpi=300)
    print("Successfully rendered geometric_heatmaps.pdf")


if __name__ == "__main__":
    main()
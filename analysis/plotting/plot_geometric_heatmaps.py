#!/usr/bin/env python3
"""
analysis/plot_geometric_heatmaps.py

Layer × method heatmap of Δsr(W_eff) as a percentage of pretrained sr.
Red = geometric degradation, blue = geometric preservation/improvement.
Intended for appendix.

Reads: results/analysis/metrics_cache.json
Output: results/analysis/figures/geometric_heatmaps.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Shared style ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from analysis.plot_style import apply_style, METHOD_LABELS_SHORT

apply_style()

OUT_DIR  = Path("results/analysis/figures")
N_LAYERS = 12

# Task to display — STS-B is geometrically interesting; fall back to first available
PREFERRED_TASK = "stsb"

# Display order: PAFT variants first, then additive, then non-additive, full FT last
METHOD_ORDER = [
    "safe_hybrid_paft", "hybrid_paft", "safe_pure_paft", "pure_paft",
    "polar_r8", "lora_r64", "lora_r8",
    "bitfit", "svf",
    "full_ft", "frozen",
]


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/metrics_cache.json not found.")

    with open(cache_path) as f:
        glue = json.load(f)["glue"]

    task = PREFERRED_TASK if PREFERRED_TASK in glue else list(glue.keys())[0]
    task_data = glue[task]

    # Determine display order: use METHOD_ORDER, include only what's in the cache
    methods = [m for m in METHOD_ORDER if m in task_data]
    # Append any unexpected methods not in our order list
    for m in sorted(task_data.keys()):
        if m not in methods:
            methods.append(m)

    # ── Build matrix ──────────────────────────────────────────────────────────
    matrix = np.full((len(methods), N_LAYERS), np.nan)

    for m_idx, method in enumerate(methods):
        entry   = task_data[method]
        # NOTE: entry.get("sr_Weff_init") is the TASK-AVERAGED scalar — using
        # it as the baseline for every layer's delta was the bug here. Every
        # layer has a real, different natural baseline sr even at pretrained
        # init (layer 0 vs layer 9 can differ by ~20%), so subtracting one
        # shared average conflated real training-induced change with each
        # layer's ordinary deviation from the task-wide mean. This produced
        # nonzero, layer-varying "damage" even for Frozen/BitFit, which never
        # touch W_V at all — caught via a raw-file audit that confirmed their
        # per-layer sr is bit-identical between init and final.
        per_layer = entry.get("per_layer", [])
        for layer_entry in per_layer:
            l_idx = int(layer_entry["layer"])
            if l_idx >= N_LAYERS:
                continue
            final_sr = layer_entry.get("sr_Weff_final")
            layer_init_sr = layer_entry.get("sr_Weff_init")
            if final_sr is not None and layer_init_sr is not None and layer_init_sr != 0:
                matrix[m_idx, l_idx] = (final_sr - layer_init_sr) / layer_init_sr * 100.0

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(methods) + 1.0))

    # Symmetric diverging colormap centred at 0.
    #
    # Linear vmin/vmax (the old approach) sets the color scale from a single
    # percentile across ALL methods. Since fixing the per-layer baseline bug,
    # most methods now correctly show small, near-zero deltas while pure-PAFT
    # shows a genuinely large one (~-15 to -30%) — one outlier method was
    # setting the scale for everyone else, crushing every other method's
    # real (but smaller) signal into near-white. A symmetric-log norm gives
    # linear, readable resolution near zero (where most real data lives) and
    # compresses — rather than gets dominated by — the few large outlier
    # values, so small real effects stay visible instead of disappearing.
    vmax = np.nanpercentile(np.abs(matrix), 95)
    vmax = max(vmax, 5.0)
    # linthresh: the +/- range treated as linear before switching to log.
    # Set from the data itself (smallest meaningful nonzero deltas) rather
    # than a fixed guess, so it adapts if the typical effect size changes.
    nonzero_abs = np.abs(matrix[np.isfinite(matrix) & (matrix != 0)])
    linthresh = max(float(np.nanpercentile(nonzero_abs, 25)), 0.1) if nonzero_abs.size else 1.0

    cmap = mpl.cm.RdBu_r
    cmap.set_bad("#f0f0f0")   # NaN cells in light grey

    norm = mpl.colors.SymLogNorm(
        linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10,
    )

    im = ax.imshow(
        matrix, cmap=cmap, aspect="auto", norm=norm,
    )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label(r"$\Delta sr(W_\mathrm{eff})$ (%, symmetric-log scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels(range(N_LAYERS), fontsize=7)
    ax.set_xlabel("Layer", fontsize=9)

    display_labels = [METHOD_LABELS_SHORT.get(m, m) for m in methods]
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(display_labels, fontsize=7.5)

    # Task label as a non-bold subtitle
    task_label = {"cola": "CoLA", "mrpc": "MRPC", "stsb": "STS-B",
                  "sst2": "SST-2", "qnli": "QNLI", "rte": "RTE",
                  "qqp": "QQP", "mnli": "MNLI"}.get(task, task.upper())
    ax.set_title(task_label, fontsize=9, pad=6)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "geometric_heatmaps.pdf")
    print("Saved: results/analysis/figures/geometric_heatmaps.pdf")


if __name__ == "__main__":
    main()
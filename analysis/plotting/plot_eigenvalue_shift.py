#!/usr/bin/env python3
"""
analysis/plot_eigenvalue_shift.py

Per-layer S-matrix adaptation profile for PAFT methods:
  Left y-axis:  mean |Δλ| (V and O projections) — scaling shift magnitude
  Right y-axis: S_V asymmetry ratio — how unevenly S distributes scaling

Both metrics are averaged across all 8 GLUE tasks per layer.
This figure shows that safe variants consistently apply smaller, more
symmetric scaling shifts than their unsafe counterparts.

Reads: results/analysis/paft_cache.json
Output: results/analysis/figures/eigenvalue_shift.pdf
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).parent))
from analysis.plot_style import apply_style, COLORS, MARKERS, LINESTYLES, METHOD_LABELS_SHORT, fig_double

apply_style()

OUT_DIR = Path("results/analysis/figures")

PAFT_METHODS = [
    "safe_hybrid_paft", "hybrid_paft",
    "safe_pure_paft",   "pure_paft",
]


def collect_per_layer(data: dict, method: str, field: str) -> dict[int, list[float]]:
    """Return {layer_idx: [values across tasks]} for one method and field."""
    result: dict[int, list[float]] = defaultdict(list)
    for task_data in data.values():
        if method not in task_data:
            continue
        for entry in task_data[method].get("per_layer", []):
            val = entry.get(field)
            if val is not None:
                result[int(entry["layer"])].append(float(val))
    return result


def layer_means(layer_dict: dict[int, list[float]]) -> tuple[list[int], list[float]]:
    layers = sorted(layer_dict.keys())
    means  = [float(np.mean(layer_dict[l])) for l in layers]
    return layers, means


def main():
    cache_path = Path("results/analysis/paft_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/paft_cache.json not found.")

    with open(cache_path) as f:
        data = json.load(f)

    fig, ax1 = fig_double(height_ratio=0.50)
    ax2 = ax1.twinx()

    # Remove right spine styling that apply_style() hides — twin axis needs it
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["top"].set_visible(False)

    delta_handles  = []
    asym_handles   = []

    for method in PAFT_METHODS:
        color  = COLORS.get(method, "#333")
        marker = MARKERS.get(method, "o")
        label  = METHOD_LABELS_SHORT.get(method, method)

        # ── |Δλ_V| on left axis ───────────────────────────────────────────
        d_v = collect_per_layer(data, method, "delta_lam_V_magnitude")
        if d_v:
            layers, means = layer_means(d_v)
            line, = ax1.plot(
                layers, means,
                color=color, linestyle=LINESTYLES.get(method, "-"),
                marker=marker, markersize=3.5, linewidth=1.3,
                label=label, zorder=3,
            )
            delta_handles.append(line)

        # ── S_V asymmetry ratio on right axis (dashed, same color) ────────
        a_v = collect_per_layer(data, method, "S_V_asymmetry_ratio")
        if a_v:
            layers, means = layer_means(a_v)
            line2, = ax2.plot(
                layers, means,
                color=color, linestyle=":",
                marker=marker, markersize=2.5, linewidth=0.9,
                alpha=0.7, zorder=2,
            )
            # Only add one asymmetry proxy handle for the legend
            if not asym_handles:
                asym_handles.append(
                    mlines.Line2D([], [], color="#555", linestyle=":",
                                  linewidth=0.9, label="$S_V$ asymmetry ratio")
                )

    # ── Axes labels ───────────────────────────────────────────────────────────
    ax1.set_xlabel("Layer", fontsize=9)
    ax1.set_ylabel(r"Mean $|\Delta\lambda_V|$", fontsize=9)
    ax2.set_ylabel(r"$S_V$ asymmetry ratio", fontsize=9, color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555", labelsize=7)

    ax1.set_xticks(range(12))
    ax1.set_xticklabels(range(12), fontsize=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    # Method handles from ax1, plus one asymmetry-style proxy
    all_handles = delta_handles + asym_handles
    all_labels  = [h.get_label() for h in all_handles]
    ax1.legend(
        all_handles, all_labels,
        loc="upper right", fontsize=7.5, frameon=True,
        facecolor="white", edgecolor="#cccccc",
        ncol=1, handletextpad=0.4,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "eigenvalue_shift.pdf")
    print("Saved: results/analysis/figures/eigenvalue_shift.pdf")


if __name__ == "__main__":
    main()

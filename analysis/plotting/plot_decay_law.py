#!/usr/bin/env python3
"""
analysis/plotting/plot_decay_law.py

Plots the logarithmic decay law sr(W_eff) = a - b*log(steps)
fitted to each method across GLUE tasks.

Two panels:
  Left:  data points + fitted curves for all methods
  Right: bar chart of decay rates b, grouped by method family

Reads: results/analysis/decay_law_results.json
       results/analysis/metrics_cache.json
Output: results/analysis/figures/decay_law.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from analysis.fit_decay_law import STEPS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plot_style import (apply_style, COLORS, MARKERS, LINESTYLES,
                        METHOD_LABELS_SHORT, fig_double)

apply_style()

OUT_DIR    = Path("results/analysis/figures")
CACHE_PATH = Path("results/analysis/metrics_cache.json")
DECAY_PATH = Path("results/analysis/decay_law_results.json")

TASK_LABELS = {
    "rte": "RTE", "mrpc": "MRPC", "stsb": "STS-B", "cola": "CoLA",
    "sst2": "SST-2", "qnli": "QNLI", "qqp": "QQP", "mnli": "MNLI",
}

# Methods to show on decay curve plot
METHODS_PLOT = [
    "safe_hybrid_paft",
    "hybrid_paft",
    "polar_r8",
    "lora_r8",
    "lora_r64",
    "svf",
]

# Methods to show on bar chart
METHODS_BAR = [
    "lora_r64",
    "lora_r8",
    "polar_r8",
    "hybrid_paft",
    "safe_hybrid_paft",
    "svf",
    "pure_paft",
    "safe_pure_paft",
]


def log_decay(steps: np.ndarray, a: float, b: float) -> np.ndarray:
    return a - b * np.log(steps)


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    if not DECAY_PATH.exists():
        sys.exit(
            f"Error: {DECAY_PATH} not found. "
            "Run python3 analysis/fit_decay_law.py first."
        )
    if not CACHE_PATH.exists():
        sys.exit(f"Error: {CACHE_PATH} not found.")

    with open(DECAY_PATH) as f:
        decay = json.load(f)
    with open(CACHE_PATH) as f:
        glue = json.load(f)["glue"]

    method_fits = decay["methods"]

    # ── Figure: two panels ────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(6.5, 2.8),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    fig.subplots_adjust(wspace=0.35)

    # ── Left panel: scatter + fitted curves ───────────────────────────────────
    # Pretrained baseline
    pretrained_sr = next(
        float(v["sr_Weff_init"])
        for task in glue.values()
        for v in task.values()
        if v.get("sr_Weff_init") is not None
    )
    ax_left.axhline(
        pretrained_sr,
        color="#bdbdbd", linestyle="--", linewidth=0.9,
        zorder=1, label=r"Pretrained $sr(W_0)$",
    )

    # Smooth x-range for fitted curves
    all_steps = list(STEPS.values())
    x_smooth = np.logspace(
        np.log10(min(all_steps) * 0.8),
        np.log10(max(all_steps) * 1.2),
        200,
    )

    legend_handles = []

    for method in METHODS_PLOT:
        fit = method_fits.get(method)
        color   = COLORS.get(method, "#333")
        marker  = MARKERS.get(method, "o")
        ls      = LINESTYLES.get(method, "-")
        label   = METHOD_LABELS_SHORT.get(method, method)
        is_paft = "paft" in method

        # Scatter: actual data points
        xs, ys = [], []
        for task, steps in STEPS.items():
            sr = glue.get(task, {}).get(method, {}).get("sr_Weff_final")
            if sr is not None:
                xs.append(steps)
                ys.append(float(sr))

        if not xs:
            continue

        ax_left.scatter(
            xs, ys,
            color=color, marker=marker,
            s=18 if is_paft else 14,
            edgecolors="white", linewidths=0.4,
            zorder=4, alpha=0.9,
        )

        # Fitted curve
        if fit is not None:
            y_fit = log_decay(x_smooth, fit["a"], fit["b"])
            line, = ax_left.plot(
                x_smooth, y_fit,
                color=color, linestyle=ls,
                linewidth=1.3 if is_paft else 1.0,
                alpha=1.0 if is_paft else 0.8,
                zorder=3,
                label=f"{label} ($b$={fit['b']:.2f})",
            )
            legend_handles.append(line)

    ax_left.set_xscale("log")
    ax_left.set_xlabel("Total gradient steps (log scale)", fontsize=8.5)
    ax_left.set_ylabel(r"$sr(W_{\mathrm{eff}})$", fontsize=8.5)

    # Add task name annotations at top of x-axis
    for task, steps in sorted(STEPS.items(), key=lambda x: x[1]):
        ax_left.axvline(steps, color="#e0e0e0", linewidth=0.4, zorder=0)

    # Legend moved OUT of the plot area — with 8 methods, a boxed legend
    # inside ax_left covered a large fraction of the curves it was meant
    # to label. The bar chart on the right already self-labels its bars
    # via set_yticklabels, so it doesn't need this legend at all — a
    # single shared strip below both panels keeps the whole plot area
    # clear on both sides.
    #
    # subplots_adjust(bottom=...) reserves real space below BOTH panels'
    # own x-axis labels before placing the legend — the earlier version
    # placed the legend at a fixed negative offset without reserving
    # space for it, which collided with the x-axis labels since both
    # panels' labels already sit close to the original bottom edge.
    handles, labels = ax_left.get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.35, bottom=0.32)
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, fontsize=6.3, frameon=True,
        facecolor="white", edgecolor="#cccccc",
        handletextpad=0.4, columnspacing=1.0,
    )

    # ── Right panel: bar chart of decay rates ─────────────────────────────────
    bar_methods = [m for m in METHODS_BAR if m in method_fits
                   and method_fits[m] is not None]
    bar_b    = [method_fits[m]["b"] for m in bar_methods]
    bar_r2   = [method_fits[m]["r2"] for m in bar_methods]
    bar_cols = [COLORS.get(m, "#333") for m in bar_methods]
    bar_lbls = [METHOD_LABELS_SHORT.get(m, m) for m in bar_methods]

    y_pos = np.arange(len(bar_methods))
    bars  = ax_right.barh(
        y_pos, bar_b,
        color=bar_cols, edgecolor="white",
        linewidth=0.5, height=0.65,
    )

    # Annotate bars with R²
    for i, (b_val, r2_val) in enumerate(zip(bar_b, bar_r2)):
        ax_right.text(
            b_val + 0.05, i,
            f"$R^2$={r2_val:.2f}",
            va="center", fontsize=6,
            color="#444444",
        )

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(bar_lbls, fontsize=7)
    ax_right.set_xlabel("Decay rate $b$", fontsize=8.5)
    ax_right.invert_yaxis()  # highest decay at top

    # Vertical reference line at b=0
    ax_right.axvline(0, color="#969696", linewidth=0.7, linestyle="--")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "decay_law.pdf", bbox_inches="tight")
    print("Saved: results/analysis/figures/decay_law.pdf")


if __name__ == "__main__":
    main()
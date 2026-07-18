#!/usr/bin/env python3
"""
analysis/plotting/plot_decay_law_wO.py

W_O companion to plot_decay_law.py. Same two-panel layout — scatter +
fitted curves on the left, bar chart of decay rates on the right — but for
sr(W_O,eff) instead of sr(W_V,eff), and restricted to methods whose
training mechanism actually reaches W_O (Frozen, BitFit, and LoRA are
excluded — their W_O is untouched by construction, plotting a decay curve
for a structural zero would be meaningless).

Reads: results/analysis/decay_law_results_wO.json
       results/analysis/metrics_cache.json
Output: results/analysis/figures/decay_law_wO.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.fit_decay_law_wO import STEPS as TASK_STEPS, FIT_METHODS, METHOD_LABELS
from analysis.plot_style import (apply_style, COLORS, MARKERS, LINESTYLES, fig_double)

apply_style()

OUT_DIR = Path("results/analysis/figures")
CACHE_PATH = Path("results/analysis/metrics_cache.json")
DECAY_PATH = Path("results/analysis/decay_law_results_wO.json")

TASK_LABELS = {
    "rte": "RTE", "mrpc": "MRPC", "stsb": "STS-B", "cola": "CoLA",
    "sst2": "SST-2", "qnli": "QNLI", "qqp": "QQP", "mnli": "MNLI",
}

# Methods to show on the decay curve panel — same scope as the fit itself
METHODS_PLOT = [
    "safe_hybrid_paft",
    "safe_pure_paft",
    "hybrid_paft",
    "pure_paft",
    "polar_r8",
    "svf",
]

METHODS_BAR = [
    "pure_paft",
    "hybrid_paft",
    "safe_pure_paft",
    "safe_hybrid_paft",
    "polar_r8",
    "svf",
]


def log_decay(steps: np.ndarray, a: float, b: float) -> np.ndarray:
    return a - b * np.log(steps)


def main():
    if not DECAY_PATH.exists():
        sys.exit(f"Error: {DECAY_PATH} not found. Run python3 analysis/fit_decay_law_wO.py first.")
    if not CACHE_PATH.exists():
        sys.exit(f"Error: {CACHE_PATH} not found.")

    with open(DECAY_PATH) as f:
        decay = json.load(f)
    with open(CACHE_PATH) as f:
        glue = json.load(f)["glue"]

    method_fits = decay["methods"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    fig.subplots_adjust(wspace=0.35)

    # Pretrained W_O baseline — read from any method's sr_Weff_O_init
    pretrained_sr_o = None
    for task in glue.values():
        for method_data in task.values():
            v = method_data.get("sr_Weff_O_init")
            if v is not None:
                pretrained_sr_o = float(v)
                break
        if pretrained_sr_o:
            break

    if pretrained_sr_o:
        ax_left.axhline(pretrained_sr_o, color=COLORS.get("pretrained", "#D6D3CC"),
                         linestyle="--", linewidth=0.9, zorder=1,
                         label=r"Pretrained $sr(W_{O,0})$")

    all_steps = list(TASK_STEPS.values())
    x_smooth = np.logspace(np.log10(min(all_steps) * 0.8), np.log10(max(all_steps) * 1.2), 200)

    for method in METHODS_PLOT:
        fit = method_fits.get(method)
        color = COLORS.get(method, "#333")
        marker = MARKERS.get(method, "o")
        ls = LINESTYLES.get(method, "-")
        label = METHOD_LABELS.get(method, method)
        is_hero = (method == "safe_hybrid_paft")

        xs, ys = [], []
        for task, steps in TASK_STEPS.items():
            sr = glue.get(task, {}).get(method, {}).get("sr_Weff_O_final")
            if sr is not None:
                xs.append(steps)
                ys.append(float(sr))
        if not xs:
            continue

        ax_left.scatter(xs, ys, color=color, marker=marker,
                         s=18 if is_hero else 14, edgecolors="white",
                         linewidths=0.4, zorder=4, alpha=0.9)

        if fit is not None:
            y_fit = log_decay(x_smooth, fit["a"], fit["b"])
            ax_left.plot(x_smooth, y_fit, color=color, linestyle=ls,
                         linewidth=1.6 if is_hero else 1.0,
                         alpha=1.0 if is_hero else 0.8, zorder=3,
                         label=f"{label} ($b_O$={fit['b']:.2f})")

    ax_left.set_xscale("log")
    ax_left.set_xlabel("Total gradient steps (log scale)", fontsize=8.5)
    ax_left.set_ylabel(r"$sr(W_{O,\mathrm{eff}})$", fontsize=8.5)

    for task, steps in sorted(TASK_STEPS.items(), key=lambda x: x[1]):
        ax_left.axvline(steps, color="#e0e0e0", linewidth=0.4, zorder=0)

    # Legend moved OUT of the plot area — same fix as plot_decay_law.py.
    # The bar chart on the right already self-labels via set_yticklabels.
    # subplots_adjust(bottom=...) reserves real space below both panels'
    # x-axis labels before placing the legend, avoiding the collision the
    # earlier fixed-offset version had.
    handles, labels = ax_left.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.32)
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, fontsize=6.3, frameon=True,
        facecolor="white", edgecolor="#cccccc",
        handletextpad=0.4, columnspacing=1.0,
    )

    # Right panel: bar chart of decay rates
    bar_methods = [m for m in METHODS_BAR if m in method_fits and method_fits[m] is not None]
    bar_b = [method_fits[m]["b"] for m in bar_methods]
    bar_r2 = [method_fits[m]["r2"] for m in bar_methods]
    bar_cols = [COLORS.get(m, "#333") for m in bar_methods]
    bar_lbls = [METHOD_LABELS.get(m, m) for m in bar_methods]

    y_pos = np.arange(len(bar_methods))
    ax_right.barh(y_pos, bar_b, color=bar_cols, edgecolor="white", linewidth=0.5, height=0.65)

    for i, (b_val, r2_val) in enumerate(zip(bar_b, bar_r2)):
        ax_right.text(b_val + 0.05, i, f"$R^2$={r2_val:.2f}", va="center",
                      fontsize=6, color="#444444")

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(bar_lbls, fontsize=7)
    ax_right.set_xlabel(r"Decay rate $b_O$", fontsize=8.5)
    ax_right.invert_yaxis()
    ax_right.axvline(0, color="#969696", linewidth=0.7, linestyle="--")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "decay_law_wO.pdf", bbox_inches="tight")
    print("Saved: results/analysis/figures/decay_law_wO.pdf")


if __name__ == "__main__":
    main()
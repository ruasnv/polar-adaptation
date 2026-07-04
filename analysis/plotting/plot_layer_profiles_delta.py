#!/usr/bin/env python3
"""
analysis/plot_layer_profiles_delta.py

Per-layer Δsr(W_eff) = sr_final − sr_init.
Three panels: CoLA (syntactic), MRPC (semantic), QQP (large-scale semantic).
QQP shows whether gradient step accumulation amplifies Layer 0-1 damage.

Methods (main figure): safe_hybrid_paft, hybrid_paft, polar_r8, full_ft
pure_paft excluded — set SHOW_PURE_PAFT=True for appendix version.

Reads: results/analysis/metrics_cache.json
Output: results/analysis/figures/layer_profiles_delta.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import (apply_style, COLORS, MARKERS, LINESTYLES,
                        METHOD_LABELS_SHORT)

apply_style()

OUT_DIR = Path("results/analysis/figures")

METHODS = [
    "safe_hybrid_paft",
    "hybrid_paft",
    "polar_r8",
    "full_ft",
]

# Set True to add pure_paft as thin faded line (appendix mode)
SHOW_PURE_PAFT = False

TASKS = ["cola", "mrpc", "qqp"]
TASK_LABELS = {
    "cola": "CoLA",
    "mrpc": "MRPC",
    "qqp":  "QQP",
}


def plot_panel(ax, task_data, task, add_legend_labels=False):
    """Plot one task panel. Returns True if any data was plotted."""
    init_sr = next(
        (v["sr_Weff_init"] for v in task_data.values() if "sr_Weff_init" in v),
        None,
    )
    if init_sr is None:
        return False

    # Optional pure_paft background line
    if SHOW_PURE_PAFT and "pure_paft" in task_data:
        per_layer = task_data["pure_paft"].get("per_layer", [])
        if per_layer:
            x = [l["layer"] for l in per_layer]
            y = [l["sr_Weff_final"] - init_sr for l in per_layer]
            ax.plot(x, y,
                    color=COLORS.get("pure_paft", "#3182bd"),
                    linestyle="--", linewidth=0.8, alpha=0.40,
                    label=METHOD_LABELS_SHORT.get("pure_paft") if add_legend_labels else "",
                    zorder=1)

    layer9_vals = {}
    for method in METHODS:
        if method not in task_data:
            continue
        per_layer = task_data[method].get("per_layer", [])
        if not per_layer:
            continue

        x = [l["layer"] for l in per_layer]
        y = [l["sr_Weff_final"] - init_sr for l in per_layer]

        ax.plot(
            x, y,
            color=COLORS.get(method, "#333"),
            linestyle=LINESTYLES.get(method, "-"),
            marker=MARKERS.get(method, "o"),
            markersize=3.0,
            linewidth=1.3,
            label=METHOD_LABELS_SHORT.get(method, method) if add_legend_labels else "",
            zorder=3 if "paft" in method else 2,
        )
        if len(y) > 9:
            layer9_vals[method] = y[9]

    # Pretrained baseline
    ax.axhline(0, color="#969696", linewidth=0.8, linestyle="--", zorder=1,
               label="pretrained" if add_legend_labels else "")

    # Layer 9 annotation — only if all methods exceed 0 there
    if layer9_vals and all(v > 0 for v in layer9_vals.values()):
        y_max = max(layer9_vals.values())
        ax.annotate(
            "L9",
            xy=(9, y_max),
            xytext=(9, y_max + abs(y_max) * 0.30 + 0.05),
            fontsize=6.5, color="#555555", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="#888888",
                            lw=0.7, mutation_scale=6),
        )

    ax.set_xlabel("Encoder layer", fontsize=9)
    ax.set_xticks(range(12))
    ax.set_xticklabels(range(12), fontsize=7)
    ax.grid(True, axis="y", alpha=0.2)

    # Panel label in top-left corner (no title)
    ax.text(0.03, 0.97, TASK_LABELS.get(task, task.upper()),
            transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="#252525")

    return True


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/metrics_cache.json not found.")

    with open(cache_path) as f:
        glue = json.load(f)["glue"]

    # Filter to tasks that are actually in the cache
    available_tasks = [t for t in TASKS if t in glue]
    n = len(available_tasks)
    if n == 0:
        sys.exit("Error: none of the required tasks found in cache.")

    fig, axes = plt.subplots(1, n, figsize=(6.5, 6.5 * 0.40), sharey=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.30)

    for i, (ax, task) in enumerate(zip(axes, available_tasks)):
        ok = plot_panel(ax, glue[task], task, add_legend_labels=(i == 0))
        if not ok:
            ax.set_visible(False)

    axes[0].set_ylabel(
        r"$\Delta \operatorname{sr}(W_{\mathrm{eff}})$",
        fontsize=9,
    )

    # Legend above all panels — collected from first panel only
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=len(handles), fontsize=7.5,
        frameon=True, edgecolor="#cccccc",
        handletextpad=0.4, columnspacing=0.8,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "layer_profiles_delta.pdf")
    print("Saved: results/analysis/figures/layer_profiles_delta.pdf")


if __name__ == "__main__":
    main()
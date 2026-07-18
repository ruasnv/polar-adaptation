#!/usr/bin/env python3
"""
analysis/plot_layer_profiles_delta.py

Per-layer Δsr(W_eff) = sr_final − sr_init.
Three panels: CoLA (syntactic), MRPC (semantic), QQP (large-scale semantic).
QQP shows whether gradient step accumulation amplifies Layer 0-1 damage.

Methods (main figure): safe_hybrid_paft, hybrid_paft, polar_r8, full_ft
pure_paft excluded — set SHOW_PURE_PAFT=True for appendix version.

Layout: panels are stacked VERTICALLY (3 rows, 1 column), not side by
side. With 12 x-axis tick labels per panel, a 1x3 horizontal layout
squeezed each panel into ~1/3 of the page width, making the layer axis
too cramped to read cleanly. Stacking gives every panel the full page
width; sharex=True means only the bottom panel repeats the x-axis
labels, so the figure doesn't waste space on three redundant copies of
"Encoder layer".

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


def plot_panel(ax, task_data, task, add_legend_labels=False, show_xlabel=True):
    """Plot one task panel. Returns True if any data was plotted."""
    # NOTE: previously used a single task-averaged sr_Weff_init as the
    # baseline for every layer's delta — that conflated real training-
    # induced change with each layer's natural deviation from the task-wide
    # mean (layers have real, different baseline sr even at pretrained
    # init). This produced nonzero, layer-varying lines for methods that
    # never touch W_V at all (Frozen/BitFit), caught via a raw-file audit
    # showing their per-layer sr is bit-identical between init and final.
    # Fix: use each layer's own sr_Weff_init (now stored per-layer in the
    # cache), not one shared scalar.
    has_any_data = any("sr_Weff_init" in v for v in task_data.values())
    if not has_any_data:
        return False

    # Optional pure_paft background line
    if SHOW_PURE_PAFT and "pure_paft" in task_data:
        per_layer = task_data["pure_paft"].get("per_layer", [])
        if per_layer:
            x = [l["layer"] for l in per_layer if "sr_Weff_init" in l]
            y = [l["sr_Weff_final"] - l["sr_Weff_init"] for l in per_layer if "sr_Weff_init" in l]
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

        x = [l["layer"] for l in per_layer if "sr_Weff_init" in l]
        y = [l["sr_Weff_final"] - l["sr_Weff_init"] for l in per_layer if "sr_Weff_init" in l]
        if not x:
            continue

        ax.plot(
            x, y,
            color=COLORS.get(method, "#333"),
            linestyle=LINESTYLES.get(method, "-"),
            marker=MARKERS.get(method, "o"),
            markersize=3.5,
            linewidth=1.4,
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
            fontsize=7, color="#555555", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="#888888",
                            lw=0.7, mutation_scale=6),
        )

    ax.set_xticks(range(12))
    if show_xlabel:
        ax.set_xlabel("Encoder layer", fontsize=9.5)
        ax.set_xticklabels(range(12), fontsize=8)
    else:
        # Shared x-axis — hide the tick labels on non-bottom panels rather
        # than repeating "Encoder layer" three times down the figure.
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, axis="y", alpha=0.2)

    # Panel label in top-left corner (no title)
    ax.text(0.02, 0.95, TASK_LABELS.get(task, task.upper()),
            transform=ax.transAxes,
            fontsize=9.5, va="top", ha="left", color="#252525")

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

    # Vertical stack: n rows, 1 column, full page width per panel.
    fig, axes = plt.subplots(
        n, 1, figsize=(6.5, 2.15 * n), sharex=True,
    )
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.18)

    for i, (ax, task) in enumerate(zip(axes, available_tasks)):
        is_last = (i == n - 1)
        ok = plot_panel(ax, glue[task], task, add_legend_labels=(i == 0),
                         show_xlabel=is_last)
        if not ok:
            ax.set_visible(False)
        ax.set_ylabel(
            r"$\Delta \operatorname{sr}(W_{\mathrm{eff}})$",
            fontsize=9.5,
        )

    # Legend above all panels — collected from first panel only.
    # subplots_adjust(top=...) reserves real space for the legend before
    # placement, so it sits directly against the top panel with no dead
    # gap — the earlier version computed an offset above y=1.0 without
    # reserving space for it, leaving an empty gap once bbox_inches='tight'
    # expanded the canvas to fit both the panels and the legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.88)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles), fontsize=8,
        frameon=True, edgecolor="#cccccc",
        handletextpad=0.4, columnspacing=0.9,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "layer_profiles_delta.pdf", bbox_inches="tight")
    print("Saved: results/analysis/figures/layer_profiles_delta.pdf")


if __name__ == "__main__":
    main()
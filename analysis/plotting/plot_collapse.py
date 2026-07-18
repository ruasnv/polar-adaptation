#!/usr/bin/env python3
"""
analysis/plot_collapse.py

sr(W_eff) vs training scale — one line per method, x-axis ordered by
total gradient steps (dataset_size × epochs / batch_size).

Visualises the core empirical finding: additive methods (LoRA, PoLAR)
degrade W_eff geometry monotonically with training scale, while PAFT
variants remain anchored near the pretrained baseline.

Reads: results/analysis/metrics_cache.json
Output: results/analysis/figures/collapse.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from analysis.fit_decay_law import STEPS as TASK_STEPS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.plot_style import (apply_style, COLORS, MARKERS, LINESTYLES,
                        METHOD_LABELS_SHORT, fig_double)

apply_style()

OUT_DIR = Path("results/analysis/figures")

# x-axis order: ascending gradient steps
TASK_ORDER = sorted(TASK_STEPS.keys(), key=lambda t: TASK_STEPS[t])

TASK_LABELS = {
    "cola": "CoLA", "mrpc": "MRPC",  "rte": "RTE",  "stsb": "STS-B",
    "sst2": "SST-2","qnli": "QNLI", "mnli": "MNLI", "qqp":  "QQP",
}

# Methods to show — grouped for visual clarity
METHODS = [
    # PAFT safe variants (should stay high)
    "safe_hybrid_paft",
    "safe_pure_paft",
    # PAFT base variants
    "hybrid_paft",
    "pure_paft",
    # Additive baselines (expected to decline)
    "polar_r8",
    "lora_r8",
    "lora_r64",
    # Non-additive baselines
    "bitfit",
    "svf",
    # Upper bound
    "full_ft",
]


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/metrics_cache.json not found.")

    with open(cache_path) as f:
        glue = json.load(f)["glue"]

    # Filter to tasks present in cache
    tasks = [t for t in TASK_ORDER if t in glue]
    if not tasks:
        sys.exit("Error: no tasks found in cache.")

    x_steps  = [TASK_STEPS[t] for t in tasks]
    x_labels = [TASK_LABELS[t] for t in tasks]

    fig, ax = fig_double(height_ratio=0.55)

    # ── Pretrained baseline ───────────────────────────────────────────────────
    # Read from any method's sr_Weff_init (all share the same pretrained weights)
    pretrained_sr = None
    for task in tasks:
        for method_data in glue[task].values():
            v = method_data.get("sr_Weff_init")
            if v is not None:
                pretrained_sr = float(v)
                break
        if pretrained_sr:
            break

    if pretrained_sr:
        ax.axhline(pretrained_sr, color="#bdbdbd", linestyle="--",
                   linewidth=0.9, zorder=1, label=r"Pretrained $sr(W_0)$")

    # ── Plot each method ──────────────────────────────────────────────────────
    for method in METHODS:
        ys = []
        xs = []
        for task, x in zip(tasks, x_steps):
            val = glue[task].get(method, {}).get("sr_Weff_final")
            if val is not None:
                ys.append(float(val))
                xs.append(x)

        if not ys:
            continue

        color    = COLORS.get(method, "#333")
        marker   = MARKERS.get(method, "o")
        linestyle = LINESTYLES.get(method, "-")
        label    = METHOD_LABELS_SHORT.get(method, method)
        is_paft  = "paft" in method

        ax.plot(
            xs, ys,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.0,
            linewidth=1.4 if is_paft else 1.1,
            alpha=1.0 if is_paft else 0.85,
            label=label,
            zorder=4 if is_paft else 3,
        )

        # Annotate the QQP endpoint for LoRA (the dramatic collapse point)
        if method in ("lora_r8", "lora_r64") and tasks[-1] == "qqp":
            qqp_val = glue["qqp"].get(method, {}).get("sr_Weff_final")
            if qqp_val is not None:
                ax.annotate(
                    f"{qqp_val:.1f}",
                    xy=(TASK_STEPS["qqp"], float(qqp_val)),
                    xytext=(TASK_STEPS["qqp"] * 0.72, float(qqp_val) - 1.5),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=color,
                                    lw=0.6, connectionstyle="arc3,rad=0.1"),
                )

    # ── X-axis: task labels + step counts ────────────────────────────────────
    ax.set_xticks(x_steps)
    ax.set_xticklabels(
        [f"{TASK_LABELS[t]}\n({TASK_STEPS[t]:,})" for t in tasks],
        fontsize=7,
    )
    ax.set_xscale("log")

    ax.set_xlabel("Task (total gradient steps, log scale)", fontsize=9)
    ax.set_ylabel(r"$sr(W_\mathrm{eff})$", fontsize=9)

    # ── Legend: two columns to keep it compact ────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="lower left", fontsize=7, frameon=True,
        facecolor="white", edgecolor="#cccccc",
        ncol=2, columnspacing=0.8, handletextpad=0.4,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "collapse.pdf")
    print("Saved: results/analysis/figures/collapse.pdf")


if __name__ == "__main__":
    main()
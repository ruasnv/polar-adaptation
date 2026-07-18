#!/usr/bin/env python3
"""
analysis/plot_sr_scatter.py

sr(ΔW_V) vs sr(W_eff) scatter — one point per (method, task).
Visual proof of Proposition 1: PAFT preserves geometric health
while LoRA/PoLAR degrade it proportionally to update complexity.

Output: results/analysis/figures/sr_scatter.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Shared style ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from analysis.plot_style import apply_style, COLORS, MARKERS, METHOD_LABELS

apply_style()

# ── Constants ─────────────────────────────────────────────────────────────────
# Read from cache in main() — do not hardcode
OUT_DIR       = Path("results/analysis/figures")

# Methods to include (exclude frozen — sr(ΔW) undefined, W never changes).
# full_ft included: sr(deltaW_V) is a real, measured value for it too — but
# note in any write-up that this one projection understates full_ft's true
# footprint, since its update is spread across every weight in the model,
# not concentrated in W_V the way PAFT/LoRA/SVF's updates are.
METHODS = [
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "lora_r8", "lora_r64", "polar_r8",
    "bitfit", "svf", "full_ft",
]

# Methods whose LoRA/additive sr(ΔW) sits in a very different range
# (LoRA r=64 on QQP reaches sr≈13): these go in the upper broken panel
UPPER_PANEL_THRESHOLD = 50.0   # sr(W_eff) above this → upper panel


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/metrics_cache.json not found. "
                 "Run build_cache.py first.")

    with open(cache_path) as f:
        glue = json.load(f)["glue"]

    # Read pretrained sr from cache — any method's sr_Weff_init (all share same pretrained)
    PRETRAINED_SR = next(
        float(v["sr_Weff_init"])
        for task in glue.values()
        for v in task.values()
        if v.get("sr_Weff_init") is not None
    )

    # ── Collect points ────────────────────────────────────────────────────────
    # points[method] = list of (sr_deltaW, sr_Weff) tuples, one per task
    points: dict[str, list[tuple[float, float]]] = {m: [] for m in METHODS}

    for task, methods in glue.items():
        for method in METHODS:
            if method not in methods:
                continue
            entry = methods[method]
            sr_weff  = entry.get("sr_Weff_final")
            sr_delta = entry.get("sr_deltaW_V")
            if sr_weff is None or sr_delta is None:
                continue
            # BitFit has no low-rank ΔW; its update is a bias vector → treat as sr=0
            if method == "bitfit":
                sr_delta = 0.0
            points[method].append((float(sr_delta), float(sr_weff)))

    # ── Determine y-range split ───────────────────────────────────────────────
    all_sr_weff = [y for m in points.values() for _, y in m]
    needs_break  = max(all_sr_weff) > UPPER_PANEL_THRESHOLD

    if needs_break:
        # Separate panels: top for exploded LoRA points, bottom for everything else
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(6.5, 4.5),
            gridspec_kw={"height_ratios": [1, 2.8]}
        )
        fig.subplots_adjust(hspace=0.10)
        axes = [ax_top, ax_bot]

        # Determine panel limits from data with 10% padding
        upper_vals = [y for y in all_sr_weff if y > UPPER_PANEL_THRESHOLD]
        lower_vals = [y for y in all_sr_weff if y <= UPPER_PANEL_THRESHOLD]
        pad_u = (max(upper_vals) - min(upper_vals)) * 0.15 or 5
        pad_l = (max(lower_vals) - min(lower_vals)) * 0.15 or 2
        ax_top.set_ylim(min(upper_vals) - pad_u, max(upper_vals) + pad_u)
        ax_bot.set_ylim(min(lower_vals) - pad_l, max(lower_vals) + pad_l)
    else:
        fig, ax_bot = plt.subplots(figsize=(6.5, 3.8))
        ax_top = None
        axes   = [ax_bot]

    # ── Plot points ───────────────────────────────────────────────────────────
    seen = set()
    for method, pts in points.items():
        if not pts:
            continue
        color  = COLORS.get(method, "#333333")
        marker = MARKERS.get(method, "o")
        label  = METHOD_LABELS.get(method, method)
        xs, ys = zip(*pts)

        scatter_kw = dict(
            color=color, marker=marker, s=40,
            edgecolors="white", linewidths=0.5,
            zorder=4, alpha=0.90,
            label=label if method not in seen else "",
        )
        seen.add(method)

        for ax in axes:
            ax.scatter(xs, ys, **scatter_kw)
            scatter_kw["label"] = ""   # only label once

        # Annotate extreme outliers by name (LoRA QQP collapse, safe-hybrid peak)
        for x, y in pts:
            if x < 5 and y > PRETRAINED_SR + 1:           # safe-hybrid high-health point
                ax_bot.annotate(
                    METHOD_LABELS.get(method, method).replace(" (Ours)", ""),
                    (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color=color,
                )
            if needs_break and y > UPPER_PANEL_THRESHOLD:  # exploded LoRA point
                ax_top.annotate(
                    f"{task.upper()}" if "task" in dir() else "",
                    (x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize=6, color=color,
                )

    # ── Pretrained baseline ───────────────────────────────────────────────────
    ref_kw = dict(color="#969696", linestyle="--", linewidth=0.9, zorder=2)
    ax_bot.axhline(PRETRAINED_SR, **ref_kw)
    ax_bot.text(
        ax_bot.get_xlim()[1] * 0.98 if ax_bot.get_xlim()[1] > 0 else 1,
        PRETRAINED_SR + 0.4,
        r"$sr(W_0)$", fontsize=7, color="#969696", ha="right",
    )

    # ── Broken-axis cosmetics ─────────────────────────────────────────────────
    if needs_break:
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False)

        d, kw = 0.012, dict(color="k", clip_on=False, linewidth=0.8)
        for ax, sign in [(ax_top, -1), (ax_bot, 1)]:
            y0 = 0 if sign == -1 else 1
            kw["transform"] = ax.transAxes
            ax.plot((-d, +d), (y0 - d * sign, y0 + d * sign), **kw)
            ax.plot((1 - d, 1 + d), (y0 - d * sign, y0 + d * sign), **kw)

    # ── Labels ────────────────────────────────────────────────────────────────
    ax_bot.set_xlabel(r"$sr(\Delta W_V)$", fontsize=9)
    ylabel = r"$sr(W_\mathrm{eff})$"
    if needs_break:
        fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=9)
    else:
        ax_bot.set_ylabel(ylabel, fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    # Collect handles from the bottom axis (all methods plotted there)
    handles, labels_ = ax_bot.get_legend_handles_labels()
    # Add reference line handle manually
    ref_handle = mlines.Line2D([], [], color="#969696", linestyle="--",
                               linewidth=0.9, label=r"Pretrained $sr(W_0)$")
    handles.append(ref_handle)
    labels_.append(r"Pretrained $sr(W_0)$")

    ax_bot.legend(
        handles, labels_,
        loc="lower right", fontsize=7, frameon=True,
        facecolor="white", edgecolor="#cccccc",
        ncol=2, columnspacing=0.8, handletextpad=0.4,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "sr_scatter.pdf")
    print("Saved: results/analysis/figures/sr_scatter.pdf")


if __name__ == "__main__":
    main()
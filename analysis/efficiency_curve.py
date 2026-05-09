"""
efficiency_curve.py — Pareto frontier: task performance vs parameter count.

What it produces
────────────────
1. figures/efficiency_curve/pareto_{domain}.png
   Scatter plot per domain.
   X axis: number of trainable parameters (log scale)
   Y axis: best eval_loss across all epochs (lower = better)
   Each point = one method. PAFT methods marked distinctly.
   The Pareto frontier (best loss at each parameter budget) is drawn.

2. figures/efficiency_curve/pareto_all_domains.png
   All domains as subplots in one figure for the paper.

3. figures/efficiency_curve/efficiency_table.csv
   Full table: method × domain → (n_params, best_eval_loss, best_epoch)

Core claim this figure supports
────────────────────────────────
PAFT methods (especially hybrid_paft and safe_hybrid_paft) should sit on
or near the Pareto frontier — matching LoRA performance at the same parameter
count, but without requiring a custom Riemannian optimizer.

Data source
───────────
results/checkpoints/{model}/{domain}/{method}/
    init/config.json        ← merged config (not used for params directly)
    epoch_*/metrics.json    ← eval_loss per epoch
    final/metrics.json      ← final eval_loss

Trainable parameter counts are hardcoded from the method definitions
(verified against the logged counts during training).

Usage
─────
    python analysis/efficiency_curve.py --model gpt2_small
    python scripts/run_analysis.py --analysis efficiency_curve --model gpt2_small
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader

# ──────────────────────────────────────────────────────────────────────────────
# Method metadata
# ──────────────────────────────────────────────────────────────────────────────

# Trainable parameter counts for gpt2_small (12L, 12H, d=768, d_head=64)
# These are derived from the method definitions and verified against training logs.
PARAM_COUNTS_SMALL = {
    "frozen":           0,
    "bitfit":           102_400,
    "svf":              18_432,
    "pure_paft":        18_432,
    "safe_pure_paft":   120_832,
    "lora_r8":          442_368,
    "polar":            443_904,
    "hybrid_paft":      1_179_648,
    "safe_hybrid_paft": 1_282_048,
    "lora_r64":         3_538_944,
    "full_finetune":    124_439_808,
}

# gpt2_medium (24L, 16H, d=1024, d_head=64)
PARAM_COUNTS_MEDIUM = {
    "frozen":           0,
    "bitfit":           204_800,
    "svf":              49_152,
    "pure_paft":        49_152,
    "safe_pure_paft":   253_952,
    "lora_r8":          1_048_576,
    "polar":            1_050_624,
    "hybrid_paft":      3_145_728,
    "safe_hybrid_paft": 3_350_528,
    "lora_r64":         8_388_608,
    "full_finetune":    354_823_168,
}

PARAM_COUNTS = {
    "gpt2_small":  PARAM_COUNTS_SMALL,
    "gpt2_medium": PARAM_COUNTS_MEDIUM,
}

# Method categories for plot styling
ADDITIVE_METHODS   = {"lora_r8", "lora_r64", "polar"}
PAFT_METHODS       = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
BASELINE_METHODS   = {"frozen", "bitfit", "svf", "full_finetune"}

METHOD_LABELS = {
    "frozen":           "Frozen",
    "bitfit":           "BitFit",
    "svf":              "SVF",
    "pure_paft":        "Pure PAFT",
    "safe_pure_paft":   "Safe Pure PAFT",
    "lora_r8":          "LoRA r=8",
    "polar":            "PoLAR r=8",
    "hybrid_paft":      "Hybrid PAFT",
    "safe_hybrid_paft": "Safe Hybrid PAFT",
    "lora_r64":         "LoRA r=64",
    "full_finetune":    "Full FT",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_best_eval_loss(run_dir: Path) -> Tuple[float, int]:
    """
    Return (best_eval_loss, best_epoch) across all saved epochs.
    Uses the minimum eval_loss over epochs 0..n-1.
    """
    loader = CheckpointLoader(run_dir)
    best_loss  = float("inf")
    best_epoch = 0

    for e in range(loader.n_epochs_saved()):
        try:
            metrics = loader.load_epoch_metrics(e)
            loss    = metrics.get("eval_loss", float("inf"))
            if loss < best_loss:
                best_loss  = loss
                best_epoch = e
        except FileNotFoundError:
            break

    return best_loss, best_epoch


def collect_efficiency_data(
    checkpoint_root: Path,
    model:   str,
    domains: List[str],
    methods: List[str],
) -> List[Dict]:
    """
    Collect (method, domain, n_params, best_eval_loss, best_epoch) for every
    complete run.
    """
    param_counts = PARAM_COUNTS.get(model, PARAM_COUNTS_SMALL)
    records = []

    for domain in domains:
        for method in methods:
            run_dir = checkpoint_root / model / domain / method
            if not (run_dir / "final" / "training_complete").exists():
                continue

            n_params = param_counts.get(method, -1)
            best_loss, best_epoch = load_best_eval_loss(run_dir)

            if best_loss == float("inf"):
                continue

            records.append({
                "model":       model,
                "domain":      domain,
                "method":      method,
                "n_params":    n_params,
                "best_eval_loss": round(best_loss, 4),
                "best_epoch":  best_epoch,
            })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Pareto frontier
# ──────────────────────────────────────────────────────────────────────────────

def pareto_frontier(
    points: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    """
    Compute the Pareto frontier for (n_params, eval_loss).
    We want: minimum loss at each parameter budget.
    A point is Pareto-optimal if no other point has fewer params AND lower loss.
    Returns sorted by n_params ascending.
    """
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier   = [sorted_pts[0]]
    min_loss   = sorted_pts[0][1]
    for p, l in sorted_pts[1:]:
        if l < min_loss:
            frontier.append((p, l))
            min_loss = l
    return frontier


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _method_style(method: str) -> Dict:
    if method in PAFT_METHODS:
        return {"marker": "*", "color": "#e03030", "zorder": 5, "s": 280}
    if method in ADDITIVE_METHODS:
        return {"marker": "^", "color": "#3060e0", "zorder": 4, "s": 100}
    if method == "frozen":
        return {"marker": "x", "color": "#888888", "zorder": 3, "s": 80}
    # baseline
    return {"marker": "o", "color": "#606060", "zorder": 3, "s": 80}


def plot_single_domain(
    ax,
    records: List[Dict],
    domain:  str,
    model:   str,
) -> None:
    """Draw one scatter plot on a given Axes object."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    domain_records = [r for r in records if r["domain"] == domain]
    if not domain_records:
        ax.set_title(f"{domain} (no data)")
        return

    # Scatter points
    annotated = set()
    for rec in domain_records:
        method  = rec["method"]
        n_params = max(rec["n_params"], 1)   # avoid log(0)
        loss    = rec["best_eval_loss"]
        style   = _method_style(method)
        label   = METHOD_LABELS.get(method, method)

        ax.scatter(
            np.log10(n_params), loss,
            marker = style["marker"],
            color  = style["color"],
            zorder = style["zorder"],
            s      = style["s"],
            label  = label if method not in annotated else "_nolegend_",
            edgecolors = "white", linewidths = 0.5,
        )
        annotated.add(method)

        # Annotate with method name
        ax.annotate(
            label, (np.log10(n_params), loss),
            fontsize=6, xytext=(3, 3), textcoords="offset points",
            color=style["color"],
        )

    # Pareto frontier
    pts = [(max(r["n_params"], 1), r["best_eval_loss"]) for r in domain_records]
    frontier = pareto_frontier(pts)
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.step(
            [np.log10(x) for x in fx], fy,
            where="post", color="#333333", linewidth=1.2,
            linestyle="--", alpha=0.6, label="Pareto frontier",
        )

    # Frozen baseline reference line
    frozen_recs = [r for r in domain_records if r["method"] == "frozen"]
    if frozen_recs:
        frozen_loss = frozen_recs[0]["best_eval_loss"]
        ax.axhline(
            frozen_loss, color="#888888", linewidth=0.8,
            linestyle=":", alpha=0.7, label=f"Frozen ({frozen_loss:.3f})",
        )

    ax.set_xlabel("Trainable Parameters  [log₁₀ scale]", fontsize=8)
    ax.set_ylabel("Best Eval Loss", fontsize=8)
    ax.set_title(
        f"{domain.capitalize()} — {model}",
        fontsize=9, fontweight="bold",
    )

    # X-tick labels as readable param counts
    xticks = [0, 3, 4, 5, 6, 7, 8]
    xlabels = ["0", "1K", "10K", "100K", "1M", "10M", "100M"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def plot_efficiency_curves(
    records:    List[Dict],
    figure_dir: Path,
    model:      str,
    domains:    List[str],
) -> None:
    """
    Save per-domain plots and a combined multi-panel figure.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({"font.size": 9})
    except ImportError:
        print("matplotlib not installed — skipping efficiency curve plots")
        return

    figure_dir.mkdir(parents=True, exist_ok=True)

    # Per-domain plots
    for domain in domains:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_single_domain(ax, records, domain, model)
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(),
                  loc="upper right", fontsize=7, framealpha=0.8)
        plt.tight_layout()
        fname = figure_dir / f"pareto_{domain}_{model}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # Combined multi-panel figure (paper-ready)
    if not domains:
        return
    ncols = min(len(domains), 3)
    nrows = (len(domains) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5))
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    for ax, domain in zip(axes_flat, domains):
        plot_single_domain(ax, records, domain, model)

    # Hide unused subplots
    for ax in axes_flat[len(domains):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Efficiency Curves — {model}\n"
        "* PAFT  ^ Additive  o Baseline",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = figure_dir / f"pareto_all_domains_{model}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(
    model:           str,
    domains:         List[str],
    checkpoint_root: Path,
    figure_dir:      Path,
) -> None:
    methods = list(PARAM_COUNTS_SMALL.keys())

    print(f"\n=== Efficiency Curve Analysis: {model} ===")
    print(f"Domains: {domains}")

    records = collect_efficiency_data(checkpoint_root, model, domains, methods)
    print(f"Loaded {len(records)} complete runs")

    if not records:
        print("No complete runs found — run experiments first.")
        return

    # CSV table
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / f"efficiency_table_{model}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {csv_path}")

    # Plots
    plot_efficiency_curves(records, figure_dir, model, domains)
    print(f"\nDone. Figures: {figure_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Efficiency curve / Pareto plot.")
    p.add_argument("--model",   default="gpt2_small",
                   choices=["gpt2_small", "gpt2_medium"])
    p.add_argument("--domains", nargs="+",
                   default=["news", "biomedical", "code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/efficiency_curve")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        model           = args.model,
        domains         = args.domains,
        checkpoint_root = Path(args.checkpoint_root),
        figure_dir      = Path(args.figure_dir),
    )
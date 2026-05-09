"""
geometric_health.py — Main paper figure: geometric health across all methods.

What it produces
────────────────
1. figures/geometric_health/heatmap_{metric}_{projection}.png
   One heatmap per metric (stable_rank, sv_entropy, effective_rank,
   condition_number, isotropy) per projection (W_V, W_O).
   Rows = methods, columns = layers.
   Cell colour = relative change from pretrained baseline.
   Red = degraded, blue = preserved, white = no change.

2. figures/geometric_health/summary_table.csv
   One row per (domain, method): all 6 global metrics for W_V and W_O.
   Mean and std across epochs 1-final (epoch 0 excluded — warmup distortion).

3. figures/geometric_health/preservation_scores.png
   Bar chart: single preservation score per method per domain.
   This is the condensed version of the heatmaps for the main paper table.

Data source
───────────
results/checkpoints/{model}/{domain}/{method}/
    init/geometric_health.pt    ← pretrained baseline (reference)
    epoch_*/geometric_health.pt ← per-epoch snapshots

Structure of geometric_health.pt (after _serialise_health in trainer.py):
    {
      "global":    {"W_V": {stable_rank, sv_entropy, ...}, "W_O": {...}},
      "per_layer": {0: {"W_V": {...}, "W_O": {...}}, 1: ..., ...},
      "per_head":  {0: {0: {"W_V": {...}, "W_O": {...}}, ...}, ...}
    }
    Each leaf dict has keys: stable_rank, sv_entropy, effective_rank,
    condition_number, nuclear_norm, isotropy.

Usage
─────
    python scripts/run_analysis.py --analysis geometric_health --model gpt2_small
    python analysis/geometric_health.py --model gpt2_small --domain news
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

METRICS = [
    "stable_rank",
    "sv_entropy",
    "effective_rank",
    "condition_number",
    "isotropy",
]

# Human-readable labels for plots
METRIC_LABELS = {
    "stable_rank":      "Stable Rank",
    "sv_entropy":       "SV Entropy",
    "effective_rank":   "Effective Rank",
    "condition_number": "Condition Number",
    "isotropy":         "Isotropy",
}

# For condition_number, higher = worse (reverse colour map)
# For all others, higher = better (higher rank/diversity/isotropy)
HIGHER_IS_BETTER = {
    "stable_rank":      True,
    "sv_entropy":       True,
    "effective_rank":   True,
    "condition_number": False,   # lower condition number = more stable
    "isotropy":         True,
}

PROJECTIONS = ["W_V", "W_O"]

# Method display order (cheapest to most expressive — left to right in plots)
METHOD_ORDER = [
    "frozen", "bitfit", "svf", "pure_paft", "safe_pure_paft",
    "lora_r8", "polar", "hybrid_paft", "safe_hybrid_paft",
    "lora_r64", "full_finetune",
]

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

def load_run_health(
    run_dir: Path,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Load init and all epoch geometric health snapshots for one run.

    Returns:
        (init_health, epoch_healths)
        init_health:   the pretrained baseline snapshot (or None if missing)
        epoch_healths: list of epoch snapshots in order [epoch_0, epoch_1, ...]
    """
    loader = CheckpointLoader(run_dir)

    try:
        init_health = loader.load_init_geometric_health()
    except FileNotFoundError:
        init_health = None

    epoch_healths = []
    for e in range(loader.n_epochs_saved()):
        try:
            epoch_healths.append(loader.load_epoch_geometric_health(e))
        except FileNotFoundError:
            break

    return init_health, epoch_healths


def get_global_metric(health: Dict, projection: str, metric: str) -> Optional[float]:
    """Extract one scalar metric from the global section of a health snapshot."""
    try:
        return health["global"][projection][metric]
    except (KeyError, TypeError):
        return None


def get_layer_metric(
    health: Dict, layer: int, projection: str, metric: str
) -> Optional[float]:
    """Extract one scalar metric from one layer's health snapshot."""
    try:
        return health["per_layer"][layer][projection][metric]
    except (KeyError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_relative_change(pretrained: float, adapted: float) -> float:
    """
    Relative change: (adapted - pretrained) / |pretrained|.
    Positive = increased, negative = decreased.
    For condition_number where lower is better, invert sign for plotting.
    """
    if abs(pretrained) < 1e-10:
        return 0.0
    return (adapted - pretrained) / abs(pretrained)


def compute_preservation_score(
    init_health: Dict,
    final_health: Dict,
) -> float:
    """
    Single scalar summarising how well the adapted model preserved the
    pretrained geometry.

    Score = 1 - mean_abs_relative_change across all 5 metrics × 2 projections.
    1.0 = perfect preservation, 0.0 = complete degradation.

    Condition number change is inverted so that all metrics contribute
    positively: larger change = more degradation regardless of direction.
    """
    changes = []
    for proj in PROJECTIONS:
        for metric in METRICS:
            pre = get_global_metric(init_health, proj, metric)
            post = get_global_metric(final_health, proj, metric)
            if pre is None or post is None:
                continue
            delta = compute_relative_change(pre, post)
            if not HIGHER_IS_BETTER[metric]:
                delta = -delta   # condition_number: increase = bad
            changes.append(abs(delta))

    if not changes:
        return float("nan")
    return max(0.0, 1.0 - float(np.mean(changes)))


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap: rows = methods, columns = layers
# ──────────────────────────────────────────────────────────────────────────────

def build_layer_heatmap(
    checkpoint_root: Path,
    model: str,
    domain: str,
    methods: List[str],
    projection: str,
    metric: str,
    n_layers: int = 12,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a (n_methods × n_layers) array of relative change values.

    Returns:
        data: ndarray [n_methods, n_layers]  relative changes
        valid_methods: list of methods that had data (rows in data)
    """
    data = []
    valid_methods = []

    for method in methods:
        run_dir = checkpoint_root / model / domain / method
        if not (run_dir / "final" / "training_complete").exists():
            continue

        init_health, epoch_healths = load_run_health(run_dir)
        if init_health is None or not epoch_healths:
            continue

        # Use final epoch health (most adapted state)
        final_health = epoch_healths[-1]

        row = []
        for layer in range(n_layers):
            pre  = get_layer_metric(init_health,   layer, projection, metric)
            post = get_layer_metric(final_health,  layer, projection, metric)
            if pre is None or post is None:
                row.append(0.0)
            else:
                delta = compute_relative_change(pre, post)
                if not HIGHER_IS_BETTER[metric]:
                    delta = -delta
                row.append(delta)

        data.append(row)
        valid_methods.append(method)

    if not data:
        return np.zeros((0, n_layers)), []

    return np.array(data), valid_methods


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def build_summary_table(
    checkpoint_root: Path,
    model: str,
    domains: List[str],
    methods: List[str],
) -> List[Dict]:
    """
    Build a flat list of records for the summary CSV.

    Each record: {domain, method, proj, metric, pretrained, adapted, rel_change, preservation}
    """
    records = []

    for domain in domains:
        for method in methods:
            run_dir = checkpoint_root / model / domain / method
            if not (run_dir / "final" / "training_complete").exists():
                continue

            init_health, epoch_healths = load_run_health(run_dir)
            if init_health is None or not epoch_healths:
                continue

            final_health = epoch_healths[-1]
            preservation = compute_preservation_score(init_health, final_health)

            for proj in PROJECTIONS:
                for metric in METRICS:
                    pre  = get_global_metric(init_health,  proj, metric)
                    post = get_global_metric(final_health, proj, metric)
                    if pre is None or post is None:
                        continue
                    delta = compute_relative_change(pre, post)

                    records.append({
                        "model":        model,
                        "domain":       domain,
                        "method":       method,
                        "projection":   proj,
                        "metric":       metric,
                        "pretrained":   round(pre,   4),
                        "adapted":      round(post,  4),
                        "rel_change":   round(delta, 4),
                        "preservation": round(preservation, 4),
                    })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(
    checkpoint_root: Path,
    figure_dir: Path,
    model: str,
    domain: str,
    methods: List[str],
    n_layers: int = 12,
) -> None:
    """
    Save one heatmap per (metric, projection) combination.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed — skipping heatmap plots")
        return

    figure_dir.mkdir(parents=True, exist_ok=True)

    for proj in PROJECTIONS:
        for metric in METRICS:
            data, valid_methods = build_layer_heatmap(
                checkpoint_root, model, domain, methods,
                projection=proj, metric=metric, n_layers=n_layers,
            )
            if data.shape[0] == 0:
                continue

            labels = [METHOD_LABELS.get(m, m) for m in valid_methods]

            fig, ax = plt.subplots(
                figsize=(max(8, n_layers * 0.7), max(4, len(labels) * 0.5))
            )

            # Diverging colourmap: blue = preserved/improved, red = degraded
            vmax = max(0.3, float(np.abs(data).max()))
            im = ax.imshow(
                data,
                aspect="auto",
                cmap="RdBu",
                vmin=-vmax, vmax=vmax,
                interpolation="nearest",
            )

            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)

            ax.set_xlabel("Transformer Layer")
            ax.set_title(
                f"{METRIC_LABELS[metric]} — {proj} — {domain} ({model})\n"
                "Relative change from pretrained  [blue=preserved, red=degraded]",
                fontsize=10,
            )

            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            plt.tight_layout()

            fname = figure_dir / f"heatmap_{metric}_{proj}_{domain}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")


def plot_preservation_bars(
    records: List[Dict],
    figure_dir: Path,
    model: str,
) -> None:
    """
    Bar chart: preservation score per method, grouped by domain.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure_dir.mkdir(parents=True, exist_ok=True)

    # Collect: {method: {domain: score}}
    scores: Dict[str, Dict[str, float]] = {}
    for rec in records:
        m, d, s = rec["method"], rec["domain"], rec["preservation"]
        if m not in scores:
            scores[m] = {}
        scores[m][d] = s   # will overwrite with last proj/metric — use deduplicated

    # Deduplicate by (method, domain) — take first preservation score
    seen = set()
    dedup: Dict[str, Dict[str, float]] = {}
    for rec in records:
        key = (rec["method"], rec["domain"])
        if key in seen:
            continue
        seen.add(key)
        m, d = rec["method"], rec["domain"]
        if m not in dedup:
            dedup[m] = {}
        dedup[d] = dedup.get(d, {})
        if m not in dedup:
            dedup[m] = {}
        dedup[m][d] = rec["preservation"]

    domains  = sorted({rec["domain"] for rec in records})
    methods  = [m for m in METHOD_ORDER if m in dedup]

    if not methods or not domains:
        return

    x     = np.arange(len(methods))
    width = 0.8 / max(len(domains), 1)
    colours = plt.cm.Set2(np.linspace(0, 1, len(domains)))

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 0.9), 5))

    for i, (domain, colour) in enumerate(zip(domains, colours)):
        vals = [dedup.get(m, {}).get(domain, float("nan")) for m in methods]
        ax.bar(x + i * width, vals, width, label=domain.capitalize(),
               color=colour, alpha=0.85, edgecolor="white")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="Perfect preservation")
    ax.set_xlabel("Method")
    ax.set_ylabel("Preservation Score  (1.0 = no change)")
    ax.set_title(f"Geometric Health Preservation — {model}", fontsize=11)
    ax.set_xticks(x + width * (len(domains) - 1) / 2)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    fname = figure_dir / f"preservation_scores_{model}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(
    model:            str,
    domains:          List[str],
    checkpoint_root:  Path,
    figure_dir:       Path,
    n_layers:         int = 12,
) -> None:
    methods = METHOD_ORDER

    print(f"\n=== Geometric Health Analysis: {model} ===")

    # 1. Summary table (CSV)
    print("Building summary table ...")
    records = build_summary_table(checkpoint_root, model, domains, methods)

    csv_path = figure_dir / f"summary_table_{model}.csv"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if records:
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"  Saved {csv_path}  ({len(records)} rows)")

    # 2. Heatmaps per domain
    for domain in domains:
        print(f"Plotting heatmaps: {domain} ...")
        plot_heatmaps(
            checkpoint_root, figure_dir / "heatmaps",
            model, domain, methods, n_layers,
        )

    # 3. Preservation score bar chart
    print("Plotting preservation scores ...")
    plot_preservation_bars(records, figure_dir, model)

    print(f"\nDone. Figures: {figure_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geometric health analysis.")
    p.add_argument("--model",      default="gpt2_small",
                   choices=["gpt2_small", "gpt2_medium"])
    p.add_argument("--domains",    nargs="+",
                   default=["news", "biomedical", "code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/geometric_health")
    p.add_argument("--n_layers",   type=int, default=12)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        model           = args.model,
        domains         = args.domains,
        checkpoint_root = Path(args.checkpoint_root),
        figure_dir      = Path(args.figure_dir),
        n_layers        = args.n_layers,
    )
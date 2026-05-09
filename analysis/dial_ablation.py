"""
dial_ablation.py - Rotation drift penalty ablation curves.

The dial ablation tests whether adding a soft rotation-preservation penalty
to the hybrid_paft loss changes the trade-off between task performance and
geometric health. A penalty of 0 = pure task adaptation; a large penalty
forces Q to stay near-frozen (approaching pure_paft behaviour).

Expected result: there is an optimal penalty where task performance is
maintained AND geometric health improves - the sweet spot on the dial.
If task loss is flat across penalties, the penalty is free (safe to use).

Ablation config: gpt2_small, 3 domains (news, legal, biomedical), 5 penalties
    0.0, 0.01, 0.1, 1.0, 10.0

What it produces
────────────────
figures/dial_ablation/performance_vs_penalty.png
    Line plot: eval_loss vs log(penalty) per domain.
    Shows whether the penalty hurts task performance.

figures/dial_ablation/health_vs_penalty.png
    Line plot: geometric preservation score vs log(penalty) per domain.
    Shows whether the penalty helps geometric health.

figures/dial_ablation/pareto_dial.png
    Scatter: eval_loss (x) vs preservation_score (y) per penalty per domain.
    The Pareto front shows the best achievable trade-off.

figures/dial_ablation/dial_ablation_table.csv
    (domain, penalty, best_eval_loss, preservation_score)

Usage
─────
    python analysis/dial_ablation.py --model gpt2_small
    (Run AFTER dial ablation experiments complete - Phase 3 of run plan)
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader

# Penalty values and their config method names
# Config methods: dial_ablation_{penalty with . replaced by _}
PENALTIES = [0.0, 0.01, 0.1, 1.0, 10.0]
_METHOD_NAME = {p: f"dial_ablation_{str(p).replace('.','_')}" for p in PENALTIES}

DOMAINS = ["news", "legal", "biomedical"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_best_eval_loss(run_dir: Path) -> Optional[float]:
    if not (run_dir / "final" / "training_complete").exists():
        return None
    loader = CheckpointLoader(run_dir)
    best   = float("inf")
    for e in range(loader.n_epochs_saved()):
        try:
            m = loader.load_epoch_metrics(e)
            best = min(best, m.get("eval_loss", float("inf")))
        except Exception:
            break
    return None if best == float("inf") else best


def preservation_score(run_dir: Path) -> Optional[float]:
    """Reuse inline preservation score computation."""
    loader = CheckpointLoader(run_dir)
    try:
        init_h   = loader.load_init_geometric_health()
        epoch_hs = loader.all_epoch_geometric_health()
        if not epoch_hs:
            return None
        final_h = epoch_hs[-1]
    except FileNotFoundError:
        return None

    METRICS = ["stable_rank","sv_entropy","effective_rank","condition_number","isotropy"]
    HIGHER  = {"stable_rank":True,"sv_entropy":True,"effective_rank":True,
                "condition_number":False,"isotropy":True}
    changes = []
    for proj in ("W_V","W_O"):
        for metric in METRICS:
            try:
                pre  = init_h["global"][proj][metric]
                post = final_h["global"][proj][metric]
            except (KeyError,TypeError):
                continue
            if abs(pre) < 1e-10:
                continue
            delta = (post - pre) / abs(pre)
            if not HIGHER[metric]:
                delta = -delta
            changes.append(abs(delta))
    if not changes:
        return None
    return max(0.0, 1.0 - float(np.mean(changes)))


def collect_records(checkpoint_root: Path, model: str) -> List[Dict]:
    records = []
    for domain in DOMAINS:
        for penalty in PENALTIES:
            method  = _METHOD_NAME[penalty]
            run_dir = checkpoint_root / model / domain / method
            loss    = load_best_eval_loss(run_dir)
            score   = preservation_score(run_dir) if run_dir.exists() else None
            if loss is None and score is None:
                continue
            records.append({
                "model":              model,
                "domain":             domain,
                "penalty":            penalty,
                "method":             method,
                "best_eval_loss":     round(loss,  4) if loss  is not None else "",
                "preservation_score": round(score, 4) if score is not None else "",
            })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

_DOMAIN_COLOURS = {"news":"#1f77b4","legal":"#ff7f0e","biomedical":"#2ca02c","code":"#d62728"}


def _log_x(p: float) -> float:
    """Map penalty to log-scale x value for plotting (0 → -3)."""
    return np.log10(p) if p > 0 else -3.0


def plot_curves(records: List[Dict], figure_dir: Path, model: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots"); return

    figure_dir.mkdir(parents=True, exist_ok=True)

    # ── Performance vs penalty ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for domain in DOMAINS:
        d_recs = sorted([r for r in records if r["domain"]==domain
                         and r["best_eval_loss"] != ""],
                        key=lambda r: r["penalty"])
        if not d_recs:
            continue
        xs = [_log_x(r["penalty"]) for r in d_recs]
        ys = [r["best_eval_loss"]  for r in d_recs]
        ax.plot(xs, ys, marker="o", label=domain.capitalize(),
                color=_DOMAIN_COLOURS.get(domain,"grey"), linewidth=2)
        for r, x, y in zip(d_recs, xs, ys):
            ax.annotate(f"{r['penalty']}", (x, y), fontsize=6,
                        xytext=(2,3), textcoords="offset points")

    ax.set_xlabel("Rotation Drift Penalty  [log10 scale,  −3 = penalty 0]")
    ax.set_ylabel("Best Eval Loss")
    ax.set_title(f"Task Performance vs Penalty - {model}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _annotate_xticks(ax)
    plt.tight_layout()
    fname = figure_dir / f"performance_vs_penalty_{model}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {fname}")

    # ── Geometric health vs penalty ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for domain in DOMAINS:
        d_recs = sorted([r for r in records if r["domain"]==domain
                         and r["preservation_score"] != ""],
                        key=lambda r: r["penalty"])
        if not d_recs:
            continue
        xs = [_log_x(r["penalty"])     for r in d_recs]
        ys = [r["preservation_score"]  for r in d_recs]
        ax.plot(xs, ys, marker="o", label=domain.capitalize(),
                color=_DOMAIN_COLOURS.get(domain,"grey"), linewidth=2)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="Perfect preservation")
    ax.set_xlabel("Rotation Drift Penalty  [log10 scale]")
    ax.set_ylabel("Geometric Preservation Score")
    ax.set_title(f"Geometric Health vs Penalty - {model}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _annotate_xticks(ax)
    plt.tight_layout()
    fname = figure_dir / f"health_vs_penalty_{model}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {fname}")

    # ── Pareto dial ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for domain in DOMAINS:
        d_recs = [r for r in records if r["domain"]==domain
                  and r["best_eval_loss"] != "" and r["preservation_score"] != ""]
        if not d_recs:
            continue
        xs = [r["best_eval_loss"]     for r in d_recs]
        ys = [r["preservation_score"] for r in d_recs]
        ps = [r["penalty"]            for r in d_recs]
        colour = _DOMAIN_COLOURS.get(domain, "grey")
        ax.plot(xs, ys, color=colour, linewidth=1, alpha=0.5, linestyle="--")
        sc = ax.scatter(xs, ys, c=[np.log10(p) if p>0 else -3 for p in ps],
                        cmap="plasma", s=80, zorder=5,
                        label=domain.capitalize(), edgecolors=colour, linewidths=1)
        for x, y, p in zip(xs, ys, ps):
            ax.annotate(f"penalty={p}", (x, y), fontsize=5,
                        xytext=(2,3), textcoords="offset points")

    plt.colorbar(sc, ax=ax, label="log10(penalty)  [dark=high penalty]")
    ax.set_xlabel("Best Eval Loss  (lower=better task performance)")
    ax.set_ylabel("Geometric Preservation Score  (higher=less degradation)")
    ax.set_title(f"Pareto Dial - Task vs Geometry Trade-off - {model}\n"
                 "Ideal point: lower-left corner (good task AND good geometry)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = figure_dir / f"pareto_dial_{model}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {fname}")


def _annotate_xticks(ax):
    ticks  = [-3, -2, -1, 0, 1]
    labels = ["0", "0.01", "0.1", "1", "10"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(model: str, checkpoint_root: Path, figure_dir: Path):
    print(f"\n=== Dial Ablation Analysis: {model} ===")
    print("NOTE: requires dial_ablation_* runs to be complete (Phase 3 of run plan)")

    records = collect_records(checkpoint_root, model)
    print(f"Loaded {len(records)} ablation data points")

    if not records:
        print("No dial ablation runs found - complete Phase 3 first.")
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / f"dial_ablation_table_{model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
    print(f"  Saved {csv_path}")

    plot_curves(records, figure_dir, model)
    print(f"Done. Figures: {figure_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",            default="gpt2_small")
    p.add_argument("--checkpoint_root",  default="results/checkpoints")
    p.add_argument("--figure_dir",       default="results/figures/dial_ablation")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.model, Path(a.checkpoint_root), Path(a.figure_dir))
"""
domain_correlation.py — Domain shift vs geometric degradation scatter plot.

Core claim: methods with non-additive parametrisation (PAFT) show LESS geometric
degradation as domain shift increases, while additive methods (LoRA) accumulate
proportionally more geometric damage on harder domains.

X-axis: domain shift level (pretrained GPT-2 perplexity on domain val set —
        higher perplexity = harder domain = more shift from WebText pretraining)
Y-axis: geometric preservation score (from geometric_health.py)

Each point = one (domain, method) pair.
Lines connect same method across domains.

Domain shift ranking (expected):
    news < legal < biomedical < code
    (GPT-2 was pretrained on WebText/Reddit — news is closest, code is furthest)

What it produces
────────────────
figures/domain_correlation/shift_vs_degradation.png
    Main scatter with regression lines per method group.

figures/domain_correlation/shift_vs_degradation_table.csv
    (domain, method, domain_shift_ppl, preservation_score, group)

Usage
─────
    python analysis/domain_correlation.py --model gpt2_small
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader
from paft.metrics.perplexity import eval_loss_to_ppl

ALL_METHODS = [
    "frozen","bitfit","svf","pure_paft","safe_pure_paft",
    "lora_r8","polar","hybrid_paft","safe_hybrid_paft","lora_r64","full_finetune",
]

METHOD_LABELS = {
    "frozen":"Frozen","bitfit":"BitFit","svf":"SVF",
    "pure_paft":"Pure PAFT","safe_pure_paft":"Safe Pure PAFT",
    "lora_r8":"LoRA r=8","polar":"PoLAR","hybrid_paft":"Hybrid PAFT",
    "safe_hybrid_paft":"Safe Hybrid PAFT","lora_r64":"LoRA r=64","full_finetune":"Full FT",
}

_GROUPS = {
    "paft":     {"pure_paft","hybrid_paft","safe_pure_paft","safe_hybrid_paft"},
    "additive": {"lora_r8","lora_r64","polar"},
    "baseline": {"frozen","bitfit","svf","full_finetune"},
}
_GROUP_COLOURS  = {"paft":"#d62728","additive":"#1f77b4","baseline":"#7f7f7f"}
_GROUP_MARKERS  = {"paft":"*","additive":"^","baseline":"o"}

def _group(m: str) -> str:
    for g, members in _GROUPS.items():
        if m in members:
            return g
    return "baseline"

# ──────────────────────────────────────────────────────────────────────────────
# Domain shift: pretrained perplexity on domain val set
# ──────────────────────────────────────────────────────────────────────────────

def domain_shift_ppl(
    checkpoint_root: Path,
    model: str,
    domain: str,
) -> Optional[float]:
    """
    Domain shift = frozen model's eval_loss on this domain's val set.
    The frozen model checkpoint already has this: epoch_0/metrics.json eval_loss
    (frozen model never changes, so epoch 0 eval_loss = pretrained baseline on domain).
    """
    run_dir = checkpoint_root / model / domain / "frozen"
    if not run_dir.exists():
        return None
    try:
        loader  = CheckpointLoader(run_dir)
        metrics = loader.load_epoch_metrics(0)
        return eval_loss_to_ppl(metrics["eval_loss"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Preservation score: from geometric_health checkpoints
# ──────────────────────────────────────────────────────────────────────────────

def preservation_score(run_dir: Path) -> Optional[float]:
    """
    Compute the geometric preservation score for one run.
    Reuses the logic from geometric_health.py inline to avoid circular imports.
    """
    loader = CheckpointLoader(run_dir)
    try:
        init_h  = loader.load_init_geometric_health()
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


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_records(
    checkpoint_root: Path,
    model: str,
    domains: List[str],
) -> List[Dict]:
    records = []

    # Domain shift PPL from frozen checkpoints
    shift_ppls: Dict[str,Optional[float]] = {}
    for domain in domains:
        ppl = domain_shift_ppl(checkpoint_root, model, domain)
        shift_ppls[domain] = ppl
        if ppl:
            print(f"  Domain shift ppl [{domain}]: {ppl:.2f}")
        else:
            print(f"  Domain shift ppl [{domain}]: NOT AVAILABLE (need frozen/{domain})")

    for domain in domains:
        ppl = shift_ppls.get(domain)
        if ppl is None:
            continue
        for method in ALL_METHODS:
            run_dir = checkpoint_root / model / domain / method
            if not (run_dir / "final" / "training_complete").exists():
                continue
            score = preservation_score(run_dir)
            if score is None:
                continue
            records.append({
                "model":             model,
                "domain":            domain,
                "method":            method,
                "domain_shift_ppl":  round(ppl,   3),
                "preservation_score":round(score,  4),
                "group":             _group(method),
            })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatter(records: List[Dict], figure_dir: Path, model: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot"); return

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter points
    for method in ALL_METHODS:
        m_recs = [r for r in records if r["method"] == method]
        if not m_recs:
            continue
        xs = [r["domain_shift_ppl"]   for r in m_recs]
        ys = [r["preservation_score"] for r in m_recs]
        grp = _group(method)
        marker = _GROUP_MARKERS[grp]
        colour = _GROUP_COLOURS[grp]

        ax.scatter(xs, ys, marker=marker,
                   color=colour, s=120 if grp=="paft" else 70,
                   zorder=5 if grp=="paft" else 3,
                   label=METHOD_LABELS.get(method,method),
                   edgecolors="white", linewidths=0.5)

        # Connect points (same method across domains)
        if len(xs) > 1:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [ys[i] for i in order],
                    color=colour, linewidth=0.8, alpha=0.4, linestyle="--")

        # Annotate domain names
        for r in m_recs:
            ax.annotate(r["domain"][:3].upper(),
                        (r["domain_shift_ppl"], r["preservation_score"]),
                        fontsize=5, color=colour, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")

    # Group-level regression lines
    for grp, colour in _GROUP_COLOURS.items():
        g_recs = [r for r in records if r["group"] == grp]
        if len(g_recs) < 2:
            continue
        xs = np.array([r["domain_shift_ppl"]    for r in g_recs])
        ys = np.array([r["preservation_score"]  for r in g_recs])
        coeffs = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(x_line, np.polyval(coeffs, x_line),
                color=colour, linewidth=2, alpha=0.6,
                label=f"{grp.capitalize()} trend")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":",
               alpha=0.5, label="Perfect preservation")
    ax.set_xlabel("Domain Shift (Pretrained GPT-2 Perplexity on Domain Val Set)",
                  fontsize=10)
    ax.set_ylabel("Geometric Preservation Score  (1.0 = no change)", fontsize=10)
    ax.set_title(
        f"Domain Shift vs Geometric Degradation — {model}\n"
        "* PAFT  ^ Additive  o Baseline  (dashed=trend per group)",
        fontsize=10,
    )

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=2,
              loc="lower left", framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = figure_dir / f"shift_vs_degradation_{model}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(model: str, domains: List[str], checkpoint_root: Path, figure_dir: Path):
    print(f"\n=== Domain Correlation Analysis: {model} ===")
    print("NOTE: requires frozen/{domain} to be complete for domain shift PPL.")

    records = collect_records(checkpoint_root, model, domains)
    print(f"Collected {len(records)} (domain, method) data points")

    if not records:
        print("No data — ensure frozen runs are complete for all domains.")
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / f"shift_vs_degradation_table_{model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
    print(f"  Saved {csv_path}")

    plot_scatter(records, figure_dir, model)
    print(f"Done. Figures: {figure_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2_small")
    p.add_argument("--domains", nargs="+", default=["news","legal","biomedical","code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/domain_correlation")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.model, a.domains, Path(a.checkpoint_root), Path(a.figure_dir))
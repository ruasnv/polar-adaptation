"""
layer_profiles.py — Per-layer adaptation magnitude across epochs (PAFT only).

Shows where in the network adaptation happens: early layers, middle, or late?
Hypothesis: semantic adaptation concentrates in middle layers (6-10 in GPT-2 small)
where higher-level representations live.

Metric: ||S_t - S_0||_F per layer per epoch, averaged over heads.
Uses paft_snapshot.pt (which contains S_V and S_O per epoch).

What it produces
────────────────
figures/layer_profiles/adaptation_profile_{method}_{domain}.png
    Line plot: one line per epoch, x=layer, y=||ΔS||_F.
    Shows which layers adapted most AND how that pattern evolved.

figures/layer_profiles/final_profile_all_methods_{domain}.png
    All PAFT methods at final epoch on one plot for comparison.

figures/layer_profiles/layer_profile_table_{model}.csv
    (model, domain, method, epoch, layer, delta_S_V_norm, delta_S_O_norm)
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader

PAFT_METHODS = ["pure_paft","hybrid_paft","safe_pure_paft","safe_hybrid_paft"]
METHOD_LABELS = {
    "pure_paft":"Pure PAFT","hybrid_paft":"Hybrid PAFT",
    "safe_pure_paft":"Safe Pure PAFT","safe_hybrid_paft":"Safe Hybrid PAFT",
}


def adaptation_norms_per_layer(
    S_init: List[torch.Tensor],   # [n_layers] of [H, d, d]
    S_now:  List[torch.Tensor],
) -> np.ndarray:
    """||S_t - S_0||_F per layer, mean over heads. Returns [n_layers]."""
    result = []
    for l in range(min(len(S_init), len(S_now))):
        S0 = S_init[l].float()   # [H, d, d]
        St = S_now[l].float()
        per_head = (St - S0).reshape(St.shape[0], -1).norm(dim=1)  # [H]
        result.append(per_head.mean().item())
    return np.array(result)


def collect_records(
    checkpoint_root: Path, model: str, domains: List[str]
) -> List[Dict]:
    records = []
    for domain in domains:
        for method in PAFT_METHODS:
            run_dir = checkpoint_root / model / domain / method
            if not (run_dir / "final" / "training_complete").exists():
                continue
            loader = CheckpointLoader(run_dir)
            decomp = loader.load_decomp_init()
            if decomp is None:
                continue
            S_V_0 = decomp.get("S_V_0", [])
            S_O_0 = decomp.get("S_O_0", [])
            for epoch in range(loader.n_epochs_saved()):
                snap = loader.load_epoch_paft_snapshot(epoch)
                if snap is None:
                    continue
                S_V_t = snap.get("S_V", [])
                S_O_t = snap.get("S_O", [])
                if not S_V_t or not S_V_0:
                    continue
                norms_V = adaptation_norms_per_layer(S_V_0, S_V_t)
                norms_O = adaptation_norms_per_layer(S_O_0, S_O_t)
                for l in range(len(norms_V)):
                    records.append({
                        "model": model, "domain": domain, "method": method,
                        "epoch": epoch, "layer": l,
                        "delta_S_V": round(float(norms_V[l]), 6),
                        "delta_S_O": round(float(norms_O[l]), 6),
                    })
    return records


def plot_profiles(records: List[Dict], figure_dir: Path, model: str, domains: List[str]):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        # Per-method multi-epoch profiles
        for method in PAFT_METHODS:
            m_recs = [r for r in records if r["domain"]==domain and r["method"]==method]
            if not m_recs:
                continue
            epochs   = sorted(set(r["epoch"] for r in m_recs))
            n_layers = max(r["layer"] for r in m_recs) + 1
            colours  = cm.viridis(np.linspace(0.1, 0.9, len(epochs)))
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
            for proj_idx, (proj, key) in enumerate([("W_V","delta_S_V"),("W_O","delta_S_O")]):
                ax = axes[proj_idx]
                for e, col in zip(epochs, colours):
                    e_recs = sorted([r for r in m_recs if r["epoch"]==e],
                                    key=lambda r: r["layer"])
                    vals = [r[key] for r in e_recs]
                    ax.plot(range(len(vals)), vals, color=col,
                            label=f"Epoch {e}", linewidth=1.5, marker="o", markersize=3)
                ax.set_xlabel("Layer"); ax.set_ylabel("||S_t - S_0||_F  (mean over heads)")
                ax.set_title(f"{proj} Adaptation — {method}/{domain}")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = figure_dir / f"adaptation_profile_{method}_{domain}_{model}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")

        # All methods at final epoch on one plot
        _COLOURS = {"pure_paft":"#1f77b4","hybrid_paft":"#d62728",
                    "safe_pure_paft":"#2ca02c","safe_hybrid_paft":"#ff7f0e"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for proj_idx, (proj, key) in enumerate([("W_V","delta_S_V"),("W_O","delta_S_O")]):
            ax = axes[proj_idx]
            for method in PAFT_METHODS:
                final_epoch = max(
                    (r["epoch"] for r in records
                     if r["domain"]==domain and r["method"]==method), default=None
                )
                if final_epoch is None:
                    continue
                m_recs = sorted(
                    [r for r in records if r["domain"]==domain
                     and r["method"]==method and r["epoch"]==final_epoch],
                    key=lambda r: r["layer"]
                )
                vals = [r[key] for r in m_recs]
                ax.plot(range(len(vals)), vals,
                        label=METHOD_LABELS.get(method, method),
                        color=_COLOURS.get(method, "grey"),
                        linewidth=2, marker="o", markersize=4)
            ax.set_xlabel("Layer"); ax.set_ylabel("||ΔS||_F")
            ax.set_title(f"{proj} Final Adaptation — {domain} ({model})")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = figure_dir / f"final_profile_all_methods_{domain}_{model}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def run(model: str, domains: List[str], checkpoint_root: Path , figure_dir: Path):
    print(f"\n=== Layer Profiles: {model} ===")
    records = collect_records(checkpoint_root, model, domains)
    print(f"Loaded {len(records)} records")
    if not records:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / f"layer_profile_table_{model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
    print(f"  Saved {csv_path}")
    plot_profiles(records, figure_dir, model, domains)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2_small")
    p.add_argument("--domains", nargs="+", default=["news","biomedical","code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/layer_profiles")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.model, a.domains, Path(a.checkpoint_root), Path(a.figure_dir))
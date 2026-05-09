"""
geometric_audit.py — PAFT-specific: rotation drift and eigenvalue shift.

What it produces
────────────────
1. figures/geometric_audit/rotation_drift_{domain}.png
   Heatmap: rotation drift ||Q_t - Q_0||_F per layer per epoch.
   One heatmap per (method, projection, domain).
   Shows HOW MUCH the frozen rotation Q drifts — for correctly implemented
   PAFT, drift should be exactly 0 since Q is a frozen buffer. Any non-zero
   drift indicates a bug (e.g. Q was accidentally included in optimizer).

2. figures/geometric_audit/eigenvalue_shift_{domain}.png
   Heatmap: mean absolute eigenvalue shift per layer per epoch.
   Shows how the scaling magnitude S changes across training.
   Large shifts in early layers / late layers tells a story about
   which parts of the network adapt most to the domain.

3. figures/geometric_audit/eigenvalue_evolution_{method}_{domain}.png
   Line plot: top-k eigenvalue trajectories across epochs for one method.
   Shows the learning dynamics of the scaling matrix — which eigenvalues
   grow, which shrink, which stay stable.

4. figures/geometric_audit/drift_summary.csv
   Table: (method, domain, epoch, mean_drift_V, mean_drift_O,
           mean_ev_shift_V, mean_ev_shift_O) — all layers averaged.

Core claim this figure supports
────────────────────────────────
Q is frozen → rotation_drift must be 0.0 for all PAFT variants.
The eigenvalue shift shows the geometric change is entirely in S.
The layer-wise pattern shows adaptation is concentrated in specific layers
(typically middle layers for semantic content, not early/late).

Data source
───────────
results/checkpoints/{model}/{domain}/{method}/
    init/decomp_init.pt           ← Q_V_0, Q_O_0, lam_V_0, lam_O_0
    epoch_*/paft_snapshot.pt      ← Q_V, Q_O, lam_V, lam_O at each epoch

Structure of paft_snapshot.pt (saved as dict from _snapshot_to_cpu):
    {
      "Q_V":   [n_layers × Tensor[n_heads, n_embd, d_head]],
      "Q_O":   [n_layers × Tensor[n_heads, d_head, n_embd]],
      "S_V":   [n_layers × Tensor[n_heads, d_head, d_head]],
      "S_O":   [n_layers × Tensor[n_heads, d_head, d_head]],
      "EV_V":  [n_layers × Tensor[n_heads, d_head, d_head]],
      "EV_O":  [n_layers × Tensor[n_heads, d_head, d_head]],
      "lam_V": [n_layers × Tensor[n_heads, d_head]],
      "lam_O": [n_layers × Tensor[n_heads, d_head]],
    }

Structure of decomp_init.pt:
    {
      "Q_V_0", "Q_O_0", "S_V_0", "S_O_0",
      "EV_V_0", "EV_O_0", "lam_V_0", "lam_O_0",
      "W_V_init", "W_O_init",
    }

Usage
─────
    python analysis/geometric_audit.py --model gpt2_small
    python scripts/run_analysis.py --analysis geometric_audit --model gpt2_small
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PAFT_METHODS = ["pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"]

METHOD_LABELS = {
    "pure_paft":        "Pure PAFT",
    "hybrid_paft":      "Hybrid PAFT",
    "safe_pure_paft":   "Safe Pure PAFT",
    "safe_hybrid_paft": "Safe Hybrid PAFT",
}

# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def rotation_drift_per_layer(
    Q_init: List[torch.Tensor],   # [n_layers] of [n_heads, m, d]
    Q_now:  List[torch.Tensor],   # same structure
) -> np.ndarray:
    """
    Compute mean Frobenius norm ||Q_t[l] - Q_0[l]||_F averaged over heads,
    per layer.

    Returns: ndarray [n_layers]
    For correctly frozen Q, every element should be 0.0.
    Any non-zero value is a bug indicator.
    """
    n_layers = len(Q_init)
    drifts   = np.zeros(n_layers)

    for l in range(min(n_layers, len(Q_now))):
        Q0 = Q_init[l].float()   # [H, m, d]
        Qt = Q_now[l].float()
        # Per-head Frobenius norm, then mean
        diff  = (Qt - Q0).reshape(Qt.shape[0], -1)   # [H, m*d]
        norms = diff.norm(dim=1)                       # [H]
        drifts[l] = norms.mean().item()

    return drifts


def eigenvalue_shift_per_layer(
    lam_init: List[torch.Tensor],   # [n_layers] of [n_heads, d_head]
    lam_now:  List[torch.Tensor],
) -> np.ndarray:
    """
    Mean absolute eigenvalue shift per layer, averaged over heads and eigenvalues.

    Returns: ndarray [n_layers]
    """
    n_layers = len(lam_init)
    shifts   = np.zeros(n_layers)

    for l in range(min(n_layers, len(lam_now))):
        lam0  = lam_init[l].float()   # [H, d]
        lamt  = lam_now[l].float()
        delta = (lamt - lam0).abs()   # [H, d]
        shifts[l] = delta.mean().item()

    return shifts


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_paft_audit_data(
    run_dir: Path,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Load decomp_init and all epoch PAFT snapshots for one PAFT run.

    Returns:
        (decomp_init, epoch_snapshots)
        decomp_init:      the t=0 decomposition tensors
        epoch_snapshots:  list of paft_snapshot dicts per epoch
    """
    loader = CheckpointLoader(run_dir)

    decomp_init = loader.load_decomp_init()
    if decomp_init is None:
        return None, []

    snapshots = []
    for e in range(loader.n_epochs_saved()):
        snap = loader.load_epoch_paft_snapshot(e)
        if snap is not None:
            snapshots.append(snap)

    return decomp_init, snapshots


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def build_audit_records(
    checkpoint_root: Path,
    model:   str,
    domains: List[str],
    methods: List[str] = PAFT_METHODS,
) -> List[Dict]:
    """
    Build flat records: (model, domain, method, epoch, layer,
                         drift_V, drift_O, ev_shift_V, ev_shift_O)
    """
    records = []

    for domain in domains:
        for method in methods:
            run_dir = checkpoint_root / model / domain / method
            if not (run_dir / "final" / "training_complete").exists():
                continue

            decomp_init, snapshots = load_paft_audit_data(run_dir)
            if decomp_init is None or not snapshots:
                continue

            Q_V_0   = decomp_init.get("Q_V_0",   [])
            Q_O_0   = decomp_init.get("Q_O_0",   [])
            lam_V_0 = decomp_init.get("lam_V_0", [])
            lam_O_0 = decomp_init.get("lam_O_0", [])

            for epoch, snap in enumerate(snapshots):
                Q_V_t   = snap.get("Q_V",   [])
                Q_O_t   = snap.get("Q_O",   [])
                lam_V_t = snap.get("lam_V", [])
                lam_O_t = snap.get("lam_O", [])

                if not Q_V_t or not Q_V_0:
                    continue

                drift_V  = rotation_drift_per_layer(Q_V_0, Q_V_t)
                drift_O  = rotation_drift_per_layer(Q_O_0, Q_O_t)
                shift_V  = eigenvalue_shift_per_layer(lam_V_0, lam_V_t)
                shift_O  = eigenvalue_shift_per_layer(lam_O_0, lam_O_t)

                n_layers = len(drift_V)
                for l in range(n_layers):
                    records.append({
                        "model":       model,
                        "domain":      domain,
                        "method":      method,
                        "epoch":       epoch,
                        "layer":       l,
                        "drift_V":     round(float(drift_V[l]),  6),
                        "drift_O":     round(float(drift_O[l]),  6),
                        "ev_shift_V":  round(float(shift_V[l]),  6),
                        "ev_shift_O":  round(float(shift_O[l]),  6),
                    })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_drift_heatmap(
    records:    List[Dict],
    figure_dir: Path,
    model:      str,
    domain:     str,
    key:        str,    # "drift_V" | "drift_O" | "ev_shift_V" | "ev_shift_O"
    title:      str,
) -> None:
    """
    Heatmap: rows = epochs, columns = layers, one subplot per method.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    domain_records = [r for r in records if r["domain"] == domain]
    methods = [m for m in PAFT_METHODS if any(r["method"] == m for r in domain_records)]

    if not methods:
        return

    n_layers  = max((r["layer"] for r in domain_records), default=11) + 1
    n_epochs  = max((r["epoch"] for r in domain_records), default=4)  + 1

    ncols = len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 3.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        m_recs = [r for r in domain_records if r["method"] == method]
        data   = np.full((n_epochs, n_layers), np.nan)
        for r in m_recs:
            e, l = r["epoch"], r["layer"]
            if e < n_epochs and l < n_layers:
                data[e, l] = r[key]

        vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1e-6
        im   = ax.imshow(
            data, aspect="auto", cmap="YlOrRd",
            vmin=0, vmax=max(vmax, 1e-8), interpolation="nearest",
        )
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=9)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_yticks(range(n_epochs))
        ax.set_yticklabels([f"E{e}" for e in range(n_epochs)], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.05)

    fig.suptitle(
        f"{title} — {domain} ({model})\n"
        "Rows=epochs, Cols=layers  [brighter = larger change]",
        fontsize=9,
    )
    plt.tight_layout()
    fname = figure_dir / f"{key}_{domain}_{model}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_eigenvalue_evolution(
    checkpoint_root: Path,
    figure_dir:      Path,
    model:  str,
    domain: str,
    method: str,
    top_k:  int = 8,
    n_layers_to_show: int = 4,
) -> None:
    """
    Line plot: top-k eigenvalue trajectories across epochs for selected layers.
    Shows HOW the eigenvalues move during training.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    run_dir = checkpoint_root / model / domain / method
    if not (run_dir / "final" / "training_complete").exists():
        return

    decomp_init, snapshots = load_paft_audit_data(run_dir)
    if decomp_init is None or not snapshots:
        return

    lam_V_0 = decomp_init.get("lam_V_0", [])   # [n_layers] of [H, d_head]
    if not lam_V_0:
        return

    n_layers = len(lam_V_0)
    # Show evenly-spaced layers
    layer_indices = np.linspace(0, n_layers - 1, n_layers_to_show, dtype=int)

    n_epochs = len(snapshots)
    fig, axes = plt.subplots(
        1, n_layers_to_show,
        figsize=(n_layers_to_show * 3.5, 3.5),
        sharey=False,
    )
    if n_layers_to_show == 1:
        axes = [axes]

    colours = plt.cm.viridis(np.linspace(0, 1, top_k))

    for ax, layer_idx in zip(axes, layer_indices):
        # Mean eigenvalues across heads: [n_epochs+1, d_head]
        lam_init = lam_V_0[layer_idx].float().mean(0).numpy()  # [d_head]
        lam_traj = [lam_init]

        for snap in snapshots:
            lam_t = snap.get("lam_V", [])
            if len(lam_t) > layer_idx:
                lam_traj.append(
                    lam_t[layer_idx].float().mean(0).numpy()
                )

        lam_traj = np.array(lam_traj)   # [n_epochs+1, d_head]
        epochs_x = np.arange(len(lam_traj)) - 1   # -1 = init, 0..n = epochs

        for k in range(min(top_k, lam_traj.shape[1])):
            ax.plot(
                epochs_x, lam_traj[:, k],
                color=colours[k], linewidth=1.2,
                marker="o", markersize=3,
                label=f"λ{k+1}" if k < 3 else None,
            )

        ax.axvline(-0.5, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.set_title(f"Layer {layer_idx}", fontsize=8)
        ax.set_xlabel("Epoch  (-1=init)", fontsize=7)
        ax.set_xticks(epochs_x)
        ax.set_xticklabels(
            ["init"] + [str(e) for e in range(n_epochs)], fontsize=6
        )
        ax.grid(True, alpha=0.3)
        if k < 3:
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        f"Top-{top_k} Eigenvalue Trajectories (W_V, mean over heads)\n"
        f"{METHOD_LABELS.get(method, method)} — {domain} ({model})",
        fontsize=9,
    )
    plt.tight_layout()
    fname = figure_dir / f"eigenvalue_evolution_{method}_{domain}_{model}.png"
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
    print(f"\n=== Geometric Audit (PAFT): {model} ===")

    records = build_audit_records(checkpoint_root, model, domains)
    print(f"Loaded {len(records)} layer×epoch records")

    if not records:
        print("No PAFT runs found. Run hybrid_paft or pure_paft first.")
        return

    # ── Check rotation drift (should be 0 for all PAFT) ──────────────────────
    max_drift = max(
        (r["drift_V"] for r in records), default=0.0
    )
    print(f"\nRotation drift check:")
    print(f"  Max ||Q_t - Q_0||_F across all runs: {max_drift:.2e}")
    if max_drift < 1e-4:
        print("  ✓  Q is correctly frozen — no drift detected")
    else:
        print("  ⚠  Non-zero rotation drift — Q may not be frozen correctly!")

    # ── CSV ──────────────────────────────────────────────────────────────────
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / f"drift_summary_{model}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved {csv_path}  ({len(records)} rows)")

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    heatmap_dir = figure_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    for domain in domains:
        print(f"Plotting heatmaps: {domain} ...")
        plot_drift_heatmap(
            records, heatmap_dir, model, domain,
            key="drift_V",     title="Rotation Drift Q_V  ||Q_t-Q_0||_F"
        )
        plot_drift_heatmap(
            records, heatmap_dir, model, domain,
            key="drift_O",     title="Rotation Drift Q_O  ||Q_t-Q_0||_F"
        )
        plot_drift_heatmap(
            records, heatmap_dir, model, domain,
            key="ev_shift_V",  title="Eigenvalue Shift W_V  mean|Δλ|"
        )
        plot_drift_heatmap(
            records, heatmap_dir, model, domain,
            key="ev_shift_O",  title="Eigenvalue Shift W_O  mean|Δλ|"
        )

    # ── Eigenvalue evolution per PAFT method ─────────────────────────────────
    evo_dir = figure_dir / "eigenvalue_evolution"
    evo_dir.mkdir(exist_ok=True)

    for domain in domains:
        for method in PAFT_METHODS:
            run_dir = checkpoint_root / model / domain / method
            if (run_dir / "final" / "training_complete").exists():
                print(f"Eigenvalue evolution: {method}/{domain} ...")
                plot_eigenvalue_evolution(
                    checkpoint_root, evo_dir, model, domain, method,
                )

    print(f"\nDone. Figures: {figure_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geometric audit for PAFT methods.")
    p.add_argument("--model",   default="gpt2_small",
                   choices=["gpt2_small", "gpt2_medium"])
    p.add_argument("--domains", nargs="+",
                   default=["news", "biomedical", "code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/geometric_audit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        model           = args.model,
        domains         = args.domains,
        checkpoint_root = Path(args.checkpoint_root),
        figure_dir      = Path(args.figure_dir),
    )
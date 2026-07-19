#!/usr/bin/env python3
"""
analysis/plotting/plot_training_dynamics.py

sr(W_eff) per epoch on SST-2 for all methods.

For PAFT, PoLAR, SVF, BitFit, full_ft:
    reads geometric_health.pt from epoch_N/ directories.
For LoRA:
    reads geometric_health_merged.pt from epoch_N/ directories
    (computed by scripts/compute_lora_epoch_sr.py).

Reads: results/glue/sst2/{method}/epoch_{n}/geometric_health[_merged].pt
Output: results/analysis/figures/training_dynamics.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plot_style import (apply_style, COLORS, MARKERS, LINESTYLES,
                        METHOD_LABELS, fig_double, plot_method)

apply_style()

GLUE_ROOT = Path("results/glue/sst2")
OUT_DIR   = Path("results/analysis/figures")

# Methods that use merged W_eff from epoch checkpoints
LORA_METHODS = {"lora_r8", "lora_r64"}

# All methods to attempt — frozen excluded (constant by definition)
METHODS = [
    "safe_hybrid_paft",
    "hybrid_paft",
    "safe_pure_paft",
    "pure_paft",
    "lora_r8",
    "lora_r64",
    "polar_r8",
    "svf",
    "bitfit",
    "full_ft",
]


def load_sr_from_pt(path: Path, is_merged: bool) -> float | None:
    """
    Load sr(W_V) from a geometric health checkpoint.
    Merged files have a different structure from unmerged files.
    """
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None

    if is_merged:
        # Structure: {"mean_sr_V": float, "mean_sr_O": float, ...}
        val = data.get("mean_sr_V")
        return float(val) if val is not None else None
    else:
        # Structure: {"global": {"W_V": {"V_stable_rank": float}}, ...}
        try:
            return float(data["global"]["W_V"]["V_stable_rank"])
        except (KeyError, TypeError):
            return None


def load_epoch_sr(method: str) -> tuple[list[int], list[float]]:
    """
    Load sr(W_eff) from epoch_1..N checkpoints.

    Does NOT include epoch 0 (pretrained init) as a separate point — every
    method starts from the same shared pretrained checkpoint, so an
    epoch-0 point would be numerically identical across every curve
    (redundant clutter of overlapping markers at x=0) and that same value
    is already shown persistently by the Frozen reference line across the
    whole plot. This also matches table_training_dynamics.tex, which
    starts at Epoch 1.

    Uses merged checkpoints for LoRA methods.
    """
    method_dir = GLUE_ROOT / method
    if not method_dir.is_dir():
        return [], []

    use_merged = method in LORA_METHODS
    epochs, sr_vals = [], []

    # Epoch 1..N
    epoch_dirs = sorted(
        method_dir.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1])
    )

    for ep_dir in epoch_dirs:
        n = int(ep_dir.name.split("_")[1])

        if use_merged:
            pt_path  = ep_dir / "geometric_health_merged.pt"
            is_merged = True
        else:
            pt_path   = ep_dir / "geometric_health.pt"
            is_merged = False

        sr = load_sr_from_pt(pt_path, is_merged=is_merged)
        if sr is not None:
            epochs.append(n)
            sr_vals.append(sr)
        else:
            if use_merged:
                # Try unmerged as fallback and warn
                fallback = ep_dir / "geometric_health.pt"
                sr_fb = load_sr_from_pt(fallback, is_merged=False)
                if sr_fb is not None:
                    print(
                        f"  Warning: {method} epoch {n} — "
                        f"merged not found, using unmerged "
                        f"(sr may be pretrained value)"
                    )
                    epochs.append(n)
                    sr_vals.append(sr_fb)

    return epochs, sr_vals


def main():
    if not GLUE_ROOT.exists():
        sys.exit(f"Error: {GLUE_ROOT} not found.")

    # Read pretrained sr from init checkpoint of any available method
    PRETRAINED_SR = None
    for method in METHODS:
        pt = GLUE_ROOT / method / "init" / "geometric_health.pt"
        sr = load_sr_from_pt(pt, is_merged=False)
        if sr is not None:
            PRETRAINED_SR = sr
            break
    if PRETRAINED_SR is None:
        sys.exit("Error: could not determine pretrained sr from any init checkpoint.")

    print(f"Pretrained sr(W_V) = {PRETRAINED_SR:.3f}")

    fig, ax = fig_double(height_ratio=0.52)

    plotted = []
    skipped = []

    for method in METHODS:
        epochs, sr_vals = load_epoch_sr(method)
        if not epochs:
            skipped.append(method)
            continue
        plot_method(ax, epochs, sr_vals, method, markevery=1)
        plotted.append(method)
        print(
            f"  {method:<22} epochs={epochs}  "
            f"sr: {sr_vals[0]:.2f} → {sr_vals[-1]:.2f}"
        )

    if not plotted:
        sys.exit("Error: no checkpoint data found for any method.")

    if skipped:
        print(f"\nSkipped (no checkpoints): {skipped}")

    # Pretrained reference line
    ax.axhline(
        PRETRAINED_SR,
        color="#bdbdbd", linestyle="--",
        linewidth=0.9, zorder=1,
        label=r"Pretrained $sr(W_0)$",
    )

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(r"$sr(W_\mathrm{eff})$", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Legend moved below the plot — with up to 11 entries (10 methods +
    # pretrained reference), a boxed legend inside the axes covered a
    # large fraction of the curves, especially since several lines
    # (safe-pure-PAFT, BitFit/Frozen) stay near the top of the plot for
    # the whole epoch range, right where "upper right" would sit.
    #
    # subplots_adjust(bottom=...) RESERVES real space for the legend
    # before placement, so the legend sits directly against the x-axis
    # label with no dead gap — the earlier version placed the legend at
    # an arbitrary negative figure-fraction offset without reserving
    # space for it, which left an empty gap between the axis and the
    # legend once bbox_inches='tight' expanded the canvas to fit both.
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.34)
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=4, fontsize=6.5, frameon=True,
        facecolor="white", edgecolor="#cccccc",
        columnspacing=0.8, handletextpad=0.4,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "training_dynamics.pdf", bbox_inches="tight")
    print(f"\nSaved: results/analysis/figures/training_dynamics.pdf")
    print(f"Methods plotted: {plotted}")


if __name__ == "__main__":
    main()
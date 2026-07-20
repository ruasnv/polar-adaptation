#!/usr/bin/env python3
"""
plot_lam_sigma_trajectories.py

Direct, informal test of the weight-decay hypothesis: does PAFT's lam
drift further from its pretrained-init value, proportionally, than SVF's
effective singular values do, across training epochs?

DeBERTa PAFT (pure_paft, hybrid_paft): lam_V loaded directly from
paft_snapshot.pt at each epoch (list of 12 per-layer tensors, shape
[H=12, d=64] each).

DeBERTa SVF: no raw delta_sigma is saved per epoch, so this recovers an
informal per-head singular-value proxy via fresh SVD of adapted_weights.pt
at each epoch (W_V, per-head [n,d] slices) — NOT SVF's own internal full-
matrix SVD (768x768), which would require un-reshaping back from the
stored per-head format. Good enough for a directional check, not a
bit-exact reconstruction of SVF's own parameterization.

LLaMA PAFT (pure_paft, hybrid_paft): lam / S loaded directly from
adapter.pt's flat per-layer keys.

Each trajectory is normalized to its own epoch-0(or epoch-1) value, so
PAFT's lam-based metric and SVF's singular-value-based metric are
comparable on the same relative scale despite being different quantities.

Usage:
    python3 plot_lam_sigma_trajectories.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEBERTA_TASK = "sst2"
LLAMA_TASK = "boolq"


def sorted_epoch_dirs(method_dir: Path):
    dirs = sorted(
        method_dir.glob("epoch_*"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1,
    )
    return dirs


def deberta_paft_lam_trajectory(task: str, method: str, key: str = "lam_V"):
    """Returns (epoch_numbers, mean_abs_lam_per_epoch)."""
    method_dir = Path(f"results/glue/{task}/{method}")
    epoch_dirs = sorted_epoch_dirs(method_dir)
    epochs, vals = [], []
    for ep_dir in epoch_dirs:
        snap_p = ep_dir / "paft_snapshot.pt"
        if not snap_p.exists():
            continue
        snap = torch.load(snap_p, map_location="cpu", weights_only=True)
        lam_list = snap.get(key)
        if lam_list is None:
            continue
        # lam_list: list of 12 per-layer tensors, each [H, d]
        all_vals = torch.cat([t.flatten() for t in lam_list])
        epochs.append(int(ep_dir.name.split("_")[1]))
        vals.append(float(all_vals.abs().mean().item()))
    return epochs, vals


def deberta_svf_sigma_trajectory(task: str, key: str = "W_V"):
    """
    Informal per-head SVD proxy for SVF's effective singular-value
    magnitude at each epoch. Returns (epoch_numbers, mean_sigma_per_epoch).
    """
    method_dir = Path(f"results/glue/{task}/svf")
    epoch_dirs = sorted_epoch_dirs(method_dir)
    epochs, vals = [], []
    for ep_dir in epoch_dirs:
        w_p = ep_dir / "adapted_weights.pt"
        if not w_p.exists():
            continue
        w = torch.load(w_p, map_location="cpu", weights_only=True)
        layers = w.get(key)
        if layers is None:
            continue
        head_means = []
        for layer_tensor in layers:   # [H, n, d]
            for h in range(layer_tensor.shape[0]):
                Wh = layer_tensor[h].float()
                sv = torch.linalg.svdvals(Wh)
                head_means.append(sv.mean().item())
        epochs.append(int(ep_dir.name.split("_")[1]))
        vals.append(float(np.mean(head_means)))
    return epochs, vals


def llama_paft_param_trajectory(task: str, method: str, param_name: str):
    """param_name: 'lam' for pure_paft, 'S' for hybrid_paft."""
    method_dir = Path(f"results/llama/{task}/{method}")
    epoch_dirs = sorted_epoch_dirs(method_dir)
    epochs, vals = [], []
    for ep_dir in epoch_dirs:
        adapter_p = ep_dir / "adapter.pt"
        if not adapter_p.exists():
            continue
        adapter = torch.load(adapter_p, map_location="cpu", weights_only=True)
        keys = [k for k in adapter.keys() if k.endswith(f".{param_name}")]
        if not keys:
            continue
        all_vals = torch.cat([adapter[k].flatten() for k in keys])
        epochs.append(int(ep_dir.name.split("_")[1]))
        vals.append(float(all_vals.abs().mean().item()))
    return epochs, vals


def normalize(epochs, vals):
    """Normalize to the first available epoch's value = 1.0."""
    if not vals or vals[0] == 0:
        return epochs, vals
    base = vals[0]
    return epochs, [v / base for v in vals]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel 1: DeBERTa PAFT lam vs SVF sigma, normalized ──────────────────
    ep_pure, v_pure = normalize(*deberta_paft_lam_trajectory(DEBERTA_TASK, "pure_paft"))
    ep_hybrid, v_hybrid = normalize(*deberta_paft_lam_trajectory(DEBERTA_TASK, "hybrid_paft"))
    ep_svf, v_svf = normalize(*deberta_svf_sigma_trajectory(DEBERTA_TASK))

    print(f"DeBERTa pure_paft |lam| trajectory (normalized): {list(zip(ep_pure, v_pure))}")
    print(f"DeBERTa hybrid_paft |lam| trajectory (normalized): {list(zip(ep_hybrid, v_hybrid))}")
    print(f"DeBERTa svf mean sigma trajectory (normalized, informal per-head proxy): {list(zip(ep_svf, v_svf))}")

    if v_pure:
        ax1.plot(ep_pure, v_pure, marker="o", label="pure-PAFT |lam| (normalized)", color="#CC785C")
    if v_hybrid:
        ax1.plot(ep_hybrid, v_hybrid, marker="s", label="hybrid-PAFT |lam| (normalized)", color="#A85A3F")
    if v_svf:
        ax1.plot(ep_svf, v_svf, marker="^", label="SVF mean sigma (normalized, informal)", color="#8C6BB1")
    ax1.axhline(1.0, color="#999", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Value / epoch-1 value")
    ax1.set_title(f"DeBERTa ({DEBERTA_TASK}): parameter magnitude vs. epoch")
    ax1.legend(fontsize=8)

    # ── Panel 2: LLaMA PAFT lam/S, normalized ────────────────────────────────
    ep_l_pure, v_l_pure = normalize(*llama_paft_param_trajectory(LLAMA_TASK, "pure_paft", "lam"))
    ep_l_hybrid, v_l_hybrid = normalize(*llama_paft_param_trajectory(LLAMA_TASK, "hybrid_paft", "S"))

    print(f"LLaMA pure_paft |lam| trajectory (normalized): {list(zip(ep_l_pure, v_l_pure))}")
    print(f"LLaMA hybrid_paft |S| trajectory (normalized): {list(zip(ep_l_hybrid, v_l_hybrid))}")

    if v_l_pure:
        ax2.plot(ep_l_pure, v_l_pure, marker="o", label="pure-PAFT |lam| (normalized)", color="#CC785C")
    if v_l_hybrid:
        ax2.plot(ep_l_hybrid, v_l_hybrid, marker="s", label="hybrid-PAFT |S| (normalized)", color="#A85A3F")
    ax2.axhline(1.0, color="#999", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value / epoch-0 value")
    ax2.set_title(f"LLaMA ({LLAMA_TASK}): parameter magnitude vs. epoch")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path = Path("lam_sigma_trajectories.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    print("\nWhat to look for: if the weight-decay hypothesis is right, PAFT's "
          "lam/S lines should trend visibly downward (magnitude shrinking toward "
          "zero) faster/further than SVF's sigma line, which should stay closer "
          "to 1.0 (since its trainable delta starts at zero and decay pulls it "
          "back toward that same zero, not away from a nonzero pretrained value).")


if __name__ == "__main__":
    main()
"""
residual_stability.py — Safe vs unsafe PAFT variants: residual stream norms.

Core question: Do bias terms in safe_* variants stabilise the residual stream
when adapting to large-shift domains?

The residual stream norm at layer l is proxied by the nuclear norm of W_V and W_O:
high nuclear norm → large activations propagated through that head → potentially
unstable residual stream. If safe variants reduce the nuclear norm relative to
unsafe variants in large-shift domains, it supports the residual stability claim.

Uses final/adapted_weights.pt — already saved, no model loading needed.

What it produces
────────────────
figures/residual_stability/nuclear_norm_{projection}_{domain}.png
    Line plot: per-layer nuclear norm for each method.
    Groups: safe variants vs unsafe variants vs additive methods vs frozen.

figures/residual_stability/stability_table_{model}.csv
    (domain, method, layer, W_V_nuclear_norm, W_O_nuclear_norm, W_V_isotropy, W_O_isotropy)
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader

ALL_METHODS = [
    "frozen","bitfit","svf","pure_paft","safe_pure_paft",
    "lora_r8","polar","hybrid_paft","safe_hybrid_paft","lora_r64","full_finetune",
]

_GROUPS = {
    "safe":     {"safe_pure_paft", "safe_hybrid_paft"},
    "unsafe":   {"pure_paft", "hybrid_paft", "svf"},
    "additive": {"lora_r8", "lora_r64", "polar"},
    "baseline": {"frozen", "bitfit", "full_finetune"},
}

_GROUP_COLOURS = {"safe":"#2ca02c","unsafe":"#d62728","additive":"#1f77b4","baseline":"#7f7f7f"}
_GROUP_STYLES  = {"safe":"-","unsafe":"--","additive":"-.","baseline":":"}

METHOD_LABELS = {
    "frozen":"Frozen","bitfit":"BitFit","svf":"SVF",
    "pure_paft":"Pure PAFT","safe_pure_paft":"Safe Pure PAFT",
    "lora_r8":"LoRA r=8","polar":"PoLAR","hybrid_paft":"Hybrid PAFT",
    "safe_hybrid_paft":"Safe Hybrid PAFT","lora_r64":"LoRA r=64","full_finetune":"Full FT",
}


def _group_of(method: str) -> str:
    for g, members in _GROUPS.items():
        if method in members:
            return g
    return "baseline"


def compute_layer_norms(adapted_weights: Dict) -> Dict[str, np.ndarray]:
    """
    From adapted_weights {"W_V": [n_layers×Tensor[H,n,d]], "W_O": [...]},
    compute per-layer nuclear norm (sum of singular values) averaged over heads.
    Returns {"W_V": ndarray[n_layers], "W_O": ndarray[n_layers]}
    """
    results = {}
    for proj in ("W_V", "W_O"):
        layers = adapted_weights.get(proj, [])
        norms  = []
        for W_h in layers:   # [H, m, d]
            W_h = W_h.float()
            # Per-head nuclear norm: sum of singular values
            head_norms = []
            for h in range(W_h.shape[0]):
                sv = torch.linalg.svdvals(W_h[h])
                head_norms.append(sv.sum().item())
            norms.append(np.mean(head_norms))
        results[proj] = np.array(norms)
    return results


def collect_records(
    checkpoint_root: Path,
    model: str,
    domains: List[str],
) -> List[Dict]:
    records = []
    for domain in domains:
        for method in ALL_METHODS:
            run_dir = checkpoint_root / model / domain / method
            if not (run_dir / "final" / "training_complete").exists():
                continue
            aw_path = run_dir / "final" / "adapted_weights.pt"
            if not aw_path.exists():
                continue
            aw = torch.load(aw_path, map_location="cpu", weights_only=False)
            layer_norms = compute_layer_norms(aw)
            for l, (nv, no) in enumerate(zip(layer_norms["W_V"], layer_norms["W_O"])):
                records.append({
                    "model": model, "domain": domain, "method": method,
                    "layer": l,
                    "W_V_nuclear_norm": round(float(nv), 4),
                    "W_O_nuclear_norm": round(float(no), 4),
                    "group": _group_of(method),
                })
    return records


def plot_layer_norms(
    records: List[Dict],
    figure_dir: Path,
    model: str,
    domains: List[str],
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        dom_recs = [r for r in records if r["domain"] == domain]
        if not dom_recs:
            continue
        n_layers = max(r["layer"] for r in dom_recs) + 1

        for proj in ("W_V", "W_O"):
            key = f"{proj}_nuclear_norm"
            fig, ax = plt.subplots(figsize=(9, 4))

            for method in ALL_METHODS:
                m_recs = sorted([r for r in dom_recs if r["method"] == method],
                                key=lambda r: r["layer"])
                if not m_recs:
                    continue
                vals = [r[key] for r in m_recs]
                grp  = _group_of(method)
                ax.plot(range(len(vals)), vals,
                        label=METHOD_LABELS.get(method, method),
                        color=_GROUP_COLOURS[grp],
                        linestyle=_GROUP_STYLES[grp],
                        linewidth=1.5 if grp in ("safe","unsafe") else 0.8,
                        alpha=0.9 if grp in ("safe","unsafe") else 0.5)

            ax.set_xlabel("Layer"); ax.set_ylabel("Nuclear Norm (mean over heads)")
            ax.set_title(f"{proj} Nuclear Norm per Layer — {domain} ({model})\n"
                         "safe=green, unsafe=red, additive=blue, baseline=grey")
            ax.legend(fontsize=6, ncol=3, loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = figure_dir / f"nuclear_norm_{proj}_{domain}_{model}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")


def run(model: str, domains: List[str], checkpoint_root: Path, figure_dir: Path):
    print(f"\n=== Residual Stability Analysis: {model} ===")
    records = collect_records(checkpoint_root, model, domains)
    print(f"Loaded {len(records)} layer records")
    if not records:
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / f"stability_table_{model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)
    print(f"  Saved {csv_path}")

    plot_layer_norms(records, figure_dir, model, domains)
    print(f"Done. Figures: {figure_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2_small")
    p.add_argument("--domains", nargs="+", default=["news","biomedical","code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/residual_stability")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.model, a.domains, Path(a.checkpoint_root), Path(a.figure_dir))
#!/usr/bin/env python3
"""
analysis/plot_rotation_drift.py

Q_V and Q_O Frobenius drift per layer — visual proof that PAFT's
orthogonal bases are frozen throughout training (drift = 0 everywhere).

Reads: results/analysis/paft_cache.json
  per_layer[i].Q_V_drift, per_layer[i].Q_O_drift
Output: results/analysis/figures/rotation_drift.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np

# ── Shared style ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from analysis.plot_style import apply_style, COLORS, METHOD_LABELS_SHORT, fig_double

apply_style()

OUT_DIR = Path("results/analysis/figures")

PAFT_METHODS = [
    "safe_hybrid_paft", "hybrid_paft",
    "safe_pure_paft",   "pure_paft",
]

# Line styles for V vs O projection within a method
PROJ_LS = {"V": "-", "O": "--"}


def main():
    cache_path = Path("results/analysis/paft_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/paft_cache.json not found.")

    with open(cache_path) as f:
        data = json.load(f)

    # Average drift across tasks for each method+projection+layer
    # Structure: method → proj ("V"/"O") → layer_idx → list of values across tasks
    drift: dict[str, dict[str, dict[int, list[float]]]] = {}

    for task_data in data.values():
        for method in PAFT_METHODS:
            if method not in task_data:
                continue
            drift.setdefault(method, {"V": {}, "O": {}})
            for layer_entry in task_data[method].get("per_layer", []):
                idx = int(layer_entry["layer"])
                for proj in ("V", "O"):
                    key = f"Q_{proj}_drift"
                    val = layer_entry.get(key)
                    if val is not None:
                        drift[method][proj].setdefault(idx, []).append(float(val))

    if not drift:
        sys.exit("Error: no Q_V_drift / Q_O_drift data in paft_cache.json.")

    fig, ax = fig_double(height_ratio=0.48)

    for method, projs in drift.items():
        color = COLORS.get(method, "#333")
        for proj, layer_dict in projs.items():
            if not layer_dict:
                continue
            layers   = sorted(layer_dict.keys())
            mean_drift = [float(np.mean(layer_dict[l])) for l in layers]

            ax.plot(
                layers, mean_drift,
                color=color,
                linestyle=PROJ_LS[proj],
                linewidth=1.2,
                marker="o" if proj == "V" else "s",
                markersize=3,
                label=f"{METHOD_LABELS_SHORT.get(method, method)} "
                      f"$Q_{{{proj}}}$",
                zorder=3,
            )

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel(
        r"$\|Q_\mathrm{final} - Q_\mathrm{init}\|_F$",
        fontsize=9,
    )
    ax.set_xticks(range(12))
    ax.set_xticklabels(range(12), fontsize=7)

    # If all values are numerically zero, annotate that directly
    all_vals = [
        v
        for m in drift.values()
        for p in m.values()
        for vals in p.values()
        for v in vals
    ]
    if all_vals and max(all_vals) < 1e-8:
        ax.set_ylim(-0.05, 0.5)
        ax.text(
            5.5, 0.25,
            r"All values $= 0.0$" "\n(exact invariance by construction)",
            ha="center", fontsize=8, color="#252525",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.3"),
        )

    # ── Legend: deduplicated by projection type ───────────────────────────────
    ax.legend(
        loc="upper right", fontsize=7, frameon=True,
        facecolor="white", edgecolor="#cccccc",
        ncol=2, columnspacing=0.8, handletextpad=0.4,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "rotation_drift.pdf")
    print("Saved: results/analysis/figures/rotation_drift.pdf")


if __name__ == "__main__":
    main()

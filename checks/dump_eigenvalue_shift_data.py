#!/usr/bin/env python3
"""
dump_eigenvalue_shift_data.py

eigenvalue_shift.pdf plots per-layer delta_lam_V_magnitude (primary axis)
and S_V_asymmetry_ratio (secondary axis) for the 4 PAFT variants, task-
averaged across all GLUE tasks (see plot_eigenvalue_shift.py's
collect_per_layer(), which averages over every task in paft_cache.json
for a given method+field). This data has never been printed or verified
anywhere in this project — this script pulls it directly from
paft_cache.json and reports it as a plain numerical table.

Writes to eigenvalue_shift_data.txt (also prints to terminal) — paste
into an LLM or eyeball directly against the actual figure.

Usage:
    python3 dump_eigenvalue_shift_data.py
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

CACHE_PATH = Path("../results/analysis/paft_cache.json")
OUT_PATH = Path("eigenvalue_shift_data.txt")
N_LAYERS = 12

PAFT_METHODS = [
    "safe_hybrid_paft", "hybrid_paft",
    "safe_pure_paft", "pure_paft",
]

FIELDS = ["delta_lam_V_magnitude", "S_V_asymmetry_ratio"]


def collect_per_layer(data: dict, method: str, field: str):
    """Mirrors plot_eigenvalue_shift.py's own collect_per_layer() exactly —
    averages a field across every task in paft_cache.json, per layer."""
    result = defaultdict(list)
    for task_data in data.values():
        if method not in task_data:
            continue
        for entry in task_data[method].get("per_layer", []):
            val = entry.get(field)
            if val is not None:
                result[int(entry["layer"])].append(float(val))
    return result


def main():
    if not CACHE_PATH.exists():
        msg = f"ERROR: {CACHE_PATH} not found"
        print(msg)
        OUT_PATH.write_text(msg)
        return

    with open(CACHE_PATH) as f:
        data = json.load(f)

    lines = []
    lines.append("eigenvalue_shift.pdf verification data — task-averaged per layer, "
                  "all 4 PAFT variants")
    lines.append("delta_lam_V_magnitude: mean |Δλ_V| per layer, averaged across "
                  "every GLUE task present in paft_cache.json")
    lines.append("S_V_asymmetry_ratio: mean ||S-S^T||_F/||S||_F per layer, same averaging")
    lines.append("")

    methods_missing = [m for m in PAFT_METHODS if not any(m in td for td in data.values())]
    if methods_missing:
        lines.append(f"MISSING FROM paft_cache.json ENTIRELY: {methods_missing}")
        lines.append("")

    for method in PAFT_METHODS:
        lines.append(f"── {method} ──")
        for field in FIELDS:
            layer_dict = collect_per_layer(data, method, field)
            if not layer_dict:
                lines.append(f"  {field}: NO DATA (field absent from every "
                              f"task/layer for this method)")
                continue
            missing_layers = [l for l in range(N_LAYERS) if l not in layer_dict]
            means = {l: float(np.mean(v)) for l, v in layer_dict.items()}
            n_tasks_per_layer = {l: len(v) for l, v in layer_dict.items()}
            layer_str = "  ".join(
                f"L{l:02d}={means.get(l, float('nan')):.5f}(n={n_tasks_per_layer.get(l,0)})"
                for l in range(N_LAYERS)
            )
            lines.append(f"  {field}:")
            if missing_layers:
                lines.append(f"    MISSING layers: {missing_layers}")
            lines.append(f"    {layer_str}")
        lines.append("")

    output = "\n".join(lines)
    print(output)
    OUT_PATH.write_text(output)
    print(f"\nWritten to: {OUT_PATH}")


if __name__ == "__main__":
    main()

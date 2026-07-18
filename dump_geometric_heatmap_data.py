#!/usr/bin/env python3
"""
dump_geometric_heatmap_data.py

geometric_heatmaps.pdf plots per-layer Delta-sr(%) for STS-B, all 11
methods (see geometric_heatmaps.py: PREFERRED_TASK="stsb",
METHOD_ORDER has 11 entries). No .tex table has ever exported this —
table_per_layer_cola.tex/_mrpc.tex only cover 2 tasks. But build_cache.py
builds per_layer_records for every task, not just those two, so this
data likely already exists in metrics_cache.json — this script pulls
whatever's really there and reports gaps honestly rather than assuming
full coverage.

Computes the same statistic the plot itself uses:
    Delta_sr(%) = (sr_final - sr_init) / sr_init * 100

Writes to geometric_heatmap_stsb_data.txt (also prints to terminal) —
paste into an LLM or eyeball directly against the actual heatmap.

Usage:
    python3 dump_geometric_heatmap_data.py
"""
import json
from pathlib import Path

CACHE_PATH = Path("results/analysis/metrics_cache.json")
OUT_PATH = Path("geometric_heatmap_stsb_data.txt")
TASK = "stsb"
N_LAYERS = 12

# Same order as geometric_heatmaps.py's METHOD_ORDER
METHOD_ORDER = [
    "safe_hybrid_paft", "hybrid_paft", "safe_pure_paft", "pure_paft",
    "polar_r8", "lora_r64", "lora_r8",
    "bitfit", "svf",
    "full_ft", "frozen",
]


def main():
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    task_data = cache.get("glue", {}).get(TASK, {})
    if not task_data:
        lines = [f"ERROR: no '{TASK}' data found in {CACHE_PATH}"]
        print("\n".join(lines))
        OUT_PATH.write_text("\n".join(lines))
        return

    methods_present = [m for m in METHOD_ORDER if m in task_data]
    methods_missing = [m for m in METHOD_ORDER if m not in task_data]

    lines = []
    lines.append(f"geometric_heatmaps.pdf verification data — task={TASK}, "
                  f"all methods in plot's own METHOD_ORDER")
    lines.append(f"Computed as Delta_sr(%) = (sr_final - sr_init) / sr_init * 100, "
                  f"same formula geometric_heatmaps.py itself uses.")
    lines.append("")

    if methods_missing:
        lines.append(f"MISSING FROM CACHE ENTIRELY ({len(methods_missing)}): {methods_missing}")
        lines.append("  (these methods have no results/glue/stsb/<method>/ entry at all "
                      "in metrics_cache.json — cannot verify their row in the heatmap)")
        lines.append("")

    for method in methods_present:
        entry = task_data[method]
        init_sr = entry.get("sr_Weff_init")
        per_layer = entry.get("per_layer", [])

        lines.append(f"── {method} ──")
        if init_sr is None:
            lines.append("  no sr_Weff_init — cannot compute any Delta_sr for this method")
            lines.append("")
            continue
        lines.append(f"  sr_Weff_init = {init_sr:.4f}")

        if not per_layer:
            lines.append("  per_layer is EMPTY — no per-layer data available at all "
                          "(likely no adapted_weights.pt was saved for this method; "
                          "this row cannot be verified against the heatmap)")
            lines.append("")
            continue

        row = {}
        for rec in per_layer:
            layer = rec.get("layer")
            final_sr = rec.get("sr_Weff_final")
            if layer is None or final_sr is None:
                continue
            delta_pct = (final_sr - init_sr) / init_sr * 100.0
            row[layer] = delta_pct

        missing_layers = [l for l in range(N_LAYERS) if l not in row]
        if missing_layers:
            lines.append(f"  layers present: {sorted(row.keys())}  "
                          f"MISSING layers: {missing_layers}")
        else:
            lines.append(f"  all {N_LAYERS} layers present")

        layer_str = "  ".join(
            f"L{l:02d}={row.get(l, float('nan')):+.2f}%" for l in range(N_LAYERS)
        )
        lines.append(f"  {layer_str}")
        lines.append("")

    output = "\n".join(lines)
    print(output)
    OUT_PATH.write_text(output)
    print(f"\nWritten to: {OUT_PATH}")


if __name__ == "__main__":
    main()
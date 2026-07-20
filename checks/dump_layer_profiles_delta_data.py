#!/usr/bin/env python3
"""
dump_layer_profiles_delta_data.py

Numerical dump matching layer_profiles_delta.pdf exactly: same 4 methods
(safe_hybrid_paft, hybrid_paft, polar_r8, full_ft), same 3 tasks (cola,
mrpc, qqp), same per-layer delta computation — using each layer's own
sr_Weff_init (the corrected computation), not the task-averaged scalar
that caused the earlier baseline bug.

Writes to layer_profiles_delta_data.txt (also prints to terminal).

Usage:
    python3 dump_layer_profiles_delta_data.py
"""
import json
from pathlib import Path

CACHE_PATH = Path("../results/analysis/metrics_cache.json")
OUT_PATH = Path("layer_profiles_delta_data.txt")
TASKS = ["cola", "mrpc", "qqp"]
METHODS = ["safe_hybrid_paft", "hybrid_paft", "polar_r8", "full_ft"]
N_LAYERS = 12


def main():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    glue = cache.get("glue", {})

    lines = []
    lines.append("layer_profiles_delta.pdf verification data — same 4 methods, "
                  "3 tasks as the actual figure")
    lines.append("Delta_sr = sr_final(layer) - sr_init(layer), using each "
                  "layer's OWN init value (post-fix computation).")
    lines.append("")

    for task in TASKS:
        task_data = glue.get(task, {})
        lines.append(f"{'='*70}")
        lines.append(f"TASK: {task}")
        lines.append(f"{'='*70}")
        if not task_data:
            lines.append(f"  no data for {task} in cache")
            lines.append("")
            continue

        for method in METHODS:
            entry = task_data.get(method)
            lines.append(f"── {method} ──")
            if entry is None:
                lines.append("  method not found for this task")
                lines.append("")
                continue

            per_layer = entry.get("per_layer", [])
            if not per_layer:
                lines.append("  per_layer is EMPTY")
                lines.append("")
                continue

            row = {}
            skipped = 0
            for rec in per_layer:
                layer = rec.get("layer")
                final_sr = rec.get("sr_Weff_final")
                init_sr = rec.get("sr_Weff_init")
                if layer is None or final_sr is None:
                    continue
                if init_sr is None:
                    skipped += 1
                    continue
                row[layer] = final_sr - init_sr

            if skipped:
                lines.append(f"  WARNING: {skipped} layer(s) missing per-layer "
                             f"sr_Weff_init — rebuild the cache with current build_cache.py")

            missing = [l for l in range(N_LAYERS) if l not in row]
            if missing:
                lines.append(f"  MISSING layers: {missing}")

            layer_str = "  ".join(
                f"L{l:02d}={row.get(l, float('nan')):+.3f}" for l in range(N_LAYERS)
            )
            lines.append(f"  {layer_str}")

            # Flag the specific "Layer 9 bump" claim directly, layer by layer
            if 8 in row and 9 in row and 10 in row:
                is_bump = row[9] > row[8] and row[9] > row[10]
                lines.append(f"  Layer 9 vs neighbors (L8={row[8]:+.3f}, "
                             f"L9={row[9]:+.3f}, L10={row[10]:+.3f}): "
                             f"{'LOCAL BUMP' if is_bump else 'no local bump'}")
            lines.append("")

    output = "\n".join(lines)
    print(output)
    OUT_PATH.write_text(output)
    print(f"\nWritten to: {OUT_PATH}")


if __name__ == "__main__":
    main()
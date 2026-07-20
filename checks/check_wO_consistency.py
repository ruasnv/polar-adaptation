#!/usr/bin/env python3
"""
check_wO_consistency.py

table_wO_metrics.tex has no pre-fix baseline to diff against — W_O
extraction was added in the same commit as the per-head-SVD fix, so
there's no "old" number to compare. The equivalent verification is:
does a fresh, independent recomputation from the raw init/final
adapted_weights.pt match what's actually stored in metrics_cache.json?

This directly tests the W_O-specific wiring in build_cache.py (has_o
detection, layer_record["sr_Weff_O_final"]/["sr_deltaW_O"] assignment,
which sits alongside but separate from the V-side code in the same loop)
rather than re-testing the already-confirmed V-side per-head math itself.

Usage:
    python3 check_wO_consistency.py --task cola --method pure_paft
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "..")
from analysis.build_cache import per_head_layer_metrics

CACHE_PATH = Path("../results/analysis/metrics_cache.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cola")
    p.add_argument("--method", default="pure_paft")
    p.add_argument("--results_dir", default="results/glue")
    args = p.parse_args()

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    entry = cache.get("glue", {}).get(args.task, {}).get(args.method, {})
    if not entry:
        sys.exit(f"No cache entry for {args.task}/{args.method}")

    method_dir = Path(args.results_dir) / args.task / args.method
    init_p = method_dir / "init" / "adapted_weights.pt"
    final_p = method_dir / "final" / "adapted_weights.pt"
    if not init_p.exists() or not final_p.exists():
        sys.exit(f"Missing adapted_weights.pt for {args.task}/{args.method}")

    w_i = torch.load(init_p, map_location="cpu", weights_only=True)
    w_f = torch.load(final_p, map_location="cpu", weights_only=True)

    if "W_O" not in w_i or "W_O" not in w_f:
        sys.exit(f"{args.task}/{args.method}: no W_O in adapted_weights.pt — "
                  f"nothing to verify (this should show as 'untouched' or "
                  f"null, not appear in table_wO_metrics.tex at all)")

    delta_ranks_o, entropies_o, eff_ranks_o, conds_o = [], [], [], []
    for layer in range(len(w_i["W_O"])):
        WO_init = w_i["W_O"][layer]
        WO_final = w_f["W_O"][layer]
        sr_dw_o, ent_o, er_o, cond_o = per_head_layer_metrics(WO_init, WO_final)
        if sr_dw_o is not None:
            delta_ranks_o.append(sr_dw_o)
        if ent_o is not None:
            entropies_o.append(ent_o)
        if er_o is not None:
            eff_ranks_o.append(er_o)
        if cond_o is not None:
            conds_o.append(cond_o)

    fresh_sr_delta_o = float(np.mean(delta_ranks_o)) if delta_ranks_o else None
    fresh_entropy_o = float(np.mean(entropies_o)) if entropies_o else None
    fresh_eff_rank_o = float(np.mean(eff_ranks_o)) if eff_ranks_o else None
    fresh_cond_o = float(np.mean(conds_o)) if conds_o else None

    cache_sr_delta_o = entry.get("sr_deltaW_O")
    cache_entropy_o = entry.get("spectral_entropy_Weff_O_final")
    cache_eff_rank_o = entry.get("effective_rank_Weff_O_final")
    cache_cond_o = entry.get("condition_number_O_final")

    print(f"{args.task}/{args.method} — W_O: cache value vs fresh recomputation\n")
    rows = [
        ("sr_deltaW_O", cache_sr_delta_o, fresh_sr_delta_o),
        ("spectral_entropy_O", cache_entropy_o, fresh_entropy_o),
        ("effective_rank_O", cache_eff_rank_o, fresh_eff_rank_o),
        ("condition_number_O", cache_cond_o, fresh_cond_o),
    ]

    all_match = True
    for name, cache_val, fresh_val in rows:
        if cache_val is None and fresh_val is None:
            status = "both null — OK"
        elif cache_val is None or fresh_val is None:
            status = "MISMATCH — one is null, other isn't"
            all_match = False
        elif abs(cache_val - fresh_val) < 1e-4:
            status = "MATCH"
        else:
            status = f"MISMATCH — diff={abs(cache_val - fresh_val):.6f}"
            all_match = False
        print(f"  {name:<22} cache={cache_val}  fresh={fresh_val}  [{status}]")

    print()
    if all_match:
        print("VERDICT: cache is fully consistent with a fresh recomputation. "
              "table_wO_metrics.tex's numbers for this entry are reproducible "
              "from source, not stale or miswired.")
    else:
        print("VERDICT: MISMATCH found — the cache does not match a fresh "
              "recomputation. Rerun build_cache.py (with --force if needed) "
              "and regenerate table_wO_metrics.tex before trusting this entry.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
check_per_head_fix.py

Diagnostic: compares OLD (flattened multi-head SVD) vs NEW (per-head SVD,
averaged) computation of entropy/effective-rank/condition-number, per layer,
for one task/method. Run from the project root.

Usage:
    python3 check_per_head_fix.py                        # defaults: cola / pure_paft
    python3 check_per_head_fix.py --task cola --method pure_paft
    python3 check_per_head_fix.py --task sst2 --method safe_pure_paft
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.build_cache import compute_metrics_from_weight, per_head_layer_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cola")
    p.add_argument("--method", default="pure_paft")
    p.add_argument("--results_dir", default="results/glue")
    args = p.parse_args()

    init_path = Path(args.results_dir) / args.task / args.method / "init" / "adapted_weights.pt"
    final_path = Path(args.results_dir) / args.task / args.method / "final" / "adapted_weights.pt"

    if not init_path.exists() or not final_path.exists():
        sys.exit(f"ERROR: missing {init_path} or {final_path}")

    w_i = torch.load(init_path, map_location="cpu")
    w_f = torch.load(final_path, map_location="cpu")

    print(f"\n{args.task}/{args.method} — OLD (flattened) vs NEW (per-head) per layer\n")
    header = f"{'Layer':<6}{'OLD sr_dW':>10}{'NEW sr_dW':>10}{'OLD ent':>10}{'NEW ent':>10}{'OLD er':>10}{'NEW er':>10}{'OLD cond':>12}{'NEW cond':>12}"
    print(header)
    print("-" * len(header))

    for layer in range(len(w_i["W_V"])):
        W_init = w_i["W_V"][layer]    # [H, n, d]
        W_final = w_f["W_V"][layer]

        # OLD: flatten all heads, single SVD
        W_init_flat = W_init.reshape(-1, W_init.shape[-1])
        W_final_flat = W_final.reshape(-1, W_final.shape[-1])
        dW_flat = W_final_flat - W_init_flat
        sr_dw_old, ent_old, er_old, cond_old = compute_metrics_from_weight(dW_flat)
        _, ent_old_f, er_old_f, cond_old_f = compute_metrics_from_weight(W_final_flat)

        # NEW: per-head SVD, averaged across heads
        sr_dw_new, ent_new, er_new, cond_new = per_head_layer_metrics(W_init, W_final)

        def f(v, fmt):
            return f"{v:{fmt}}" if v is not None else "N/A"

        print(f"{layer:<6}"
              f"{f(sr_dw_old, '10.4f')}{f(sr_dw_new, '10.4f')}"
              f"{f(ent_old_f, '10.4f')}{f(ent_new, '10.4f')}"
              f"{f(er_old_f, '10.4f')}{f(er_new, '10.4f')}"
              f"{f(cond_old_f, '12.3f')}{f(cond_new, '12.3f')}")

    print("\nFocus column: sr_dW (sr(deltaW_V)) — this is the field that feeds "
          "table_sr_delta_w.tex and table_stable_rank.tex's last-but-one column. "
          "If OLD/NEW here show a consistent ~3x ratio matching what you saw in "
          "the tables, that confirms the uniform shift is this exact fix, not a "
          "separate issue.")

    print("\nIf OLD/NEW entropy and eff-rank columns track closely per-layer, the "
          "task-level match to pre-fix numbers is real and expected (those metrics "
          "are not very sensitive to flatten-vs-per-head averaging). If they diverge "
          "here but the cache still shows the old aggregate, the cache wasn't "
          "actually rebuilt for this method — that's a real problem to chase down.")


if __name__ == "__main__":
    main()
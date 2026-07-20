#!/usr/bin/env python3
"""
check_negative_eigenvalues.py

Source of the "up to 2.6% of lambda_V on SST-2" claim in Method
(subsec:variants, pure-PAFT paragraph). Checks how many of pure-PAFT's
trained eigenvalues (lam_V, lam_O) have crossed zero by the final
checkpoint, for whichever task/method combinations are requested.

Usage:
    python3 check_negative_eigenvalues.py --tasks rte sst2 --method pure_paft
    python3 check_negative_eigenvalues.py --tasks cola mrpc rte stsb sst2 qnli mnli qqp --method pure_paft
"""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["rte", "sst2"])
    p.add_argument("--method", default="pure_paft")
    p.add_argument("--results_dir", default="results/glue")
    args = p.parse_args()

    for task in args.tasks:
        snap_path = Path(args.results_dir) / task / args.method / "final" / "paft_snapshot.pt"
        if not snap_path.exists():
            print(f"{task}/{args.method}: file not found ({snap_path})")
            continue

        snap = torch.load(snap_path, map_location="cpu", weights_only=True)
        for key in ("lam_V", "lam_O"):
            if key not in snap:
                continue
            all_vals = torch.cat([t.flatten() for t in snap[key]])
            n_negative = (all_vals < 0).sum().item()
            n_total = all_vals.numel()
            pct = 100.0 * n_negative / n_total
            print(f"{task}/{args.method} {key}: "
                  f"min={all_vals.min().item():.6f}  "
                  f"n_negative={n_negative}/{n_total}  "
                  f"({pct:.3f}%)")


if __name__ == "__main__":
    main()
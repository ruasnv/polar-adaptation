#!/usr/bin/env python3
"""
check_wV_reconstruction_order.py

Same test as check_wO_reconstruction_order.py, applied to the value
projection (W_V). Does NOT assume the docstring's claimed "row mode is
the reverse order of col mode" — checks both orderings directly against
the real stored W_V, including the transpose, the same way the W_O
check surfaced a real storage-convention mismatch rather than a
theoretical one.

Usage:
    python3 check_wV_reconstruction_order.py --task sst2 --method pure_paft
"""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="sst2")
    p.add_argument("--method", default="pure_paft")
    p.add_argument("--results_dir", default="results/glue")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    args = p.parse_args()

    method_dir = Path(args.results_dir) / args.task / args.method
    snap = torch.load(method_dir / "init" / "paft_snapshot.pt",
                       map_location="cpu", weights_only=True)
    weights = torch.load(method_dir / "init" / "adapted_weights.pt",
                          map_location="cpu", weights_only=True)

    Q_V = snap["Q_V"][args.layer][args.head].float()   # frozen buffer
    S_V = snap["S_V"][args.layer][args.head].float()   # trainable (at init)
    W_V_real = weights["W_V"][args.layer][args.head].float()  # actual stored weight

    print(f"Q_V shape: {tuple(Q_V.shape)}")
    print(f"S_V shape: {tuple(S_V.shape)}")
    print(f"Stored W_V (real, from adapted_weights.pt) shape: {tuple(W_V_real.shape)}")
    print()

    def try_match(name, computed):
        if computed.shape != W_V_real.shape:
            if computed.shape == W_V_real.shape[::-1]:
                diff_t = (computed - W_V_real.T).abs().max().item()
                print(f"{name}: shape {tuple(computed.shape)} matches "
                      f"TRANSPOSE of stored W_V. max|diff| vs W_V^T = {diff_t:.6e}")
                return diff_t
            print(f"{name}: shape {tuple(computed.shape)} does NOT match "
                  f"stored W_V {tuple(W_V_real.shape)} or its transpose — "
                  f"incompatible shapes")
            return None
        diff = (computed - W_V_real).abs().max().item()
        print(f"{name}: shape {tuple(computed.shape)} matches stored W_V directly. "
              f"max|diff| = {diff:.6e}")
        return diff

    print("Testing Q_V @ S_V:")
    try:
        qs = Q_V @ S_V
        d1 = try_match("  Q_V @ S_V", qs)
    except RuntimeError as e:
        print(f"  Q_V @ S_V: shape mismatch, cannot multiply ({e})")
        d1 = None

    print("Testing S_V @ Q_V:")
    try:
        sq = S_V @ Q_V
        d2 = try_match("  S_V @ Q_V", sq)
    except RuntimeError as e:
        print(f"  S_V @ Q_V: shape mismatch, cannot multiply ({e})")
        d2 = None

    print()
    print("VERDICT:")
    if d1 is not None and d1 < 1e-3:
        print("  Q_V @ S_V matches the real stored weight — forward pass computes Q0 @ M.")
    elif d2 is not None and d2 < 1e-3:
        print("  S_V @ Q_V matches the real stored weight — forward pass computes M @ Q0.")
    else:
        print("  Neither ordering matched closely — investigate further before "
              "assuming either convention.")


if __name__ == "__main__":
    main()
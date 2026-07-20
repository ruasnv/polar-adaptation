#!/usr/bin/env python3
"""
check_wO_reconstruction_order.py

Directly answers, from real checkpoint data:
  1. Is the frozen Q0 buffer for the output projection (Q_O) stored
     [n_out, d_head] (tall) or [d_head, n_out] (wide)?
  2. Does the real reconstructed W_O equal Q_O @ S_O or S_O @ Q_O?

Tests both multiplication orders against the actual stored W_O in
adapted_weights.pt, rather than trusting any docstring's description of
the intended convention.

Usage:
    python3 check_wO_reconstruction_order.py --task sst2 --method pure_paft
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

    Q_O = snap["Q_O"][args.layer][args.head].float()   # frozen buffer
    S_O = snap["S_O"][args.layer][args.head].float()   # trainable (at init)
    W_O_real = weights["W_O"][args.layer][args.head].float()  # actual stored weight

    print(f"Q_O shape: {tuple(Q_O.shape)}")
    print(f"S_O shape: {tuple(S_O.shape)}")
    print(f"Stored W_O (real, from adapted_weights.pt) shape: {tuple(W_O_real.shape)}")
    print()

    def try_match(name, computed):
        if computed.shape != W_O_real.shape:
            if computed.shape == W_O_real.shape[::-1]:
                diff_t = (computed - W_O_real.T).abs().max().item()
                print(f"{name}: shape {tuple(computed.shape)} matches "
                      f"TRANSPOSE of stored W_O. max|diff| vs W_O^T = {diff_t:.6e}")
                return diff_t
            print(f"{name}: shape {tuple(computed.shape)} does NOT match "
                  f"stored W_O {tuple(W_O_real.shape)} or its transpose — "
                  f"incompatible shapes")
            return None
        diff = (computed - W_O_real).abs().max().item()
        print(f"{name}: shape {tuple(computed.shape)} matches stored W_O directly. "
              f"max|diff| = {diff:.6e}")
        return diff

    print("Testing Q_O @ S_O:")
    try:
        qs = Q_O @ S_O
        d1 = try_match("  Q_O @ S_O", qs)
    except RuntimeError as e:
        print(f"  Q_O @ S_O: shape mismatch, cannot multiply ({e})")
        d1 = None

    print("Testing S_O @ Q_O:")
    try:
        sq = S_O @ Q_O
        d2 = try_match("  S_O @ Q_O", sq)
    except RuntimeError as e:
        print(f"  S_O @ Q_O: shape mismatch, cannot multiply ({e})")
        d2 = None

    print()
    print("VERDICT:")
    if d1 is not None and d1 < 1e-3:
        print("  Q_O @ S_O matches the real stored weight — forward pass computes Q0 @ M.")
    elif d2 is not None and d2 < 1e-3:
        print("  S_O @ Q_O matches the real stored weight — forward pass computes M @ Q0.")
    else:
        print("  Neither ordering matched closely — check whether S_O here (init) has "
              "already been updated, or whether a different reconstruction (e.g. "
              "involving EV_O separately) is actually used. Re-run with --layer/--head "
              "varied to rule out a one-off numerical issue.")


if __name__ == "__main__":
    main()
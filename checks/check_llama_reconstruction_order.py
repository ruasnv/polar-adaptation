#!/usr/bin/env python3
"""
check_llama_reconstruction_order.py

Same empirical test as the DeBERTa W_V/W_O checks, applied to LLaMA.
Checks BOTH projections even though LLaMAPAFTModel (llama_methods.py)
only wraps v_proj — Q_O/S_O keys were observed to exist in LLaMA's
paft_snapshot.pt regardless, so this checks whether they hold real
reconstructed data or are placeholder/frozen values, rather than
assuming either.

Usage:
    python3 check_llama_reconstruction_order.py --task boolq --method pure_paft
"""
import argparse
from pathlib import Path

import torch


def check_projection(proj: str, snap: dict, weights: dict, layer: int, head: int):
    print(f"\n{'='*60}\nProjection: W_{proj}\n{'='*60}")

    Q_key, S_key, W_key = f"Q_{proj}", f"S_{proj}", f"W_{proj}"

    if Q_key not in snap or S_key not in snap:
        print(f"  {Q_key}/{S_key} not present in paft_snapshot.pt — skipping")
        return
    if W_key not in weights:
        print(f"  {W_key} not present in adapted_weights.pt — skipping")
        return

    try:
        Q = snap[Q_key][layer][head].float()
        S = snap[S_key][layer][head].float()
    except (IndexError, TypeError) as e:
        print(f"  Could not index {Q_key}/{S_key} at layer={layer}, head={head}: {e}")
        print(f"  (this alone is informative — may mean these keys are placeholders "
              f"with unexpected/empty structure, not real per-layer/per-head data)")
        return

    try:
        W_real = weights[W_key][layer][head].float()
    except (IndexError, TypeError) as e:
        print(f"  Could not index {W_key} at layer={layer}, head={head}: {e}")
        return

    print(f"  Q_{proj} shape: {tuple(Q.shape)}")
    print(f"  S_{proj} shape: {tuple(S.shape)}")
    print(f"  Stored W_{proj} (real) shape: {tuple(W_real.shape)}")

    def try_match(name, computed):
        if computed.shape == W_real.shape:
            diff = (computed - W_real).abs().max().item()
            print(f"  {name}: matches stored W_{proj} directly. max|diff| = {diff:.6e}")
            return diff
        if computed.shape == W_real.shape[::-1]:
            diff = (computed - W_real.T).abs().max().item()
            print(f"  {name}: matches TRANSPOSE of stored W_{proj}. max|diff| = {diff:.6e}")
            return diff
        print(f"  {name}: shape {tuple(computed.shape)} incompatible with "
              f"stored W_{proj} {tuple(W_real.shape)} (or its transpose)")
        return None

    print(f"\n  Testing Q_{proj} @ S_{proj}:")
    try:
        d1 = try_match(f"Q_{proj} @ S_{proj}", Q @ S)
    except RuntimeError as e:
        print(f"    shape mismatch, cannot multiply ({e})")
        d1 = None

    print(f"  Testing S_{proj} @ Q_{proj}:")
    try:
        d2 = try_match(f"S_{proj} @ Q_{proj}", S @ Q)
    except RuntimeError as e:
        print(f"    shape mismatch, cannot multiply ({e})")
        d2 = None

    print(f"\n  VERDICT for W_{proj}:")
    if d1 is not None and d1 < 1e-3:
        print(f"    Q_{proj} @ S_{proj} matches — reconstruction is Q0 @ M for W_{proj}.")
    elif d2 is not None and d2 < 1e-3:
        print(f"    S_{proj} @ Q_{proj} matches — reconstruction is M @ Q0 for W_{proj}.")
    else:
        print(f"    NEITHER ordering matches the real stored weight. This means "
              f"W_{proj} in adapted_weights.pt is NOT a genuine Q/S PAFT "
              f"reconstruction — likely a frozen/untouched copy of the base "
              f"weight, consistent with LLaMA PAFT being documented as "
              f"V-only. Q_{proj}/S_{proj} existing as keys does not mean "
              f"W_{proj} was actually adapted.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="boolq")
    p.add_argument("--method", default="pure_paft")
    p.add_argument("--results_dir", default="results/llama")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    args = p.parse_args()

    method_dir = Path(args.results_dir) / args.task / args.method

    snap_path = method_dir / "init" / "paft_snapshot.pt"
    weights_path = method_dir / "init" / "adapted_weights.pt"

    if not snap_path.exists():
        print(f"ERROR: {snap_path} not found")
        return
    if not weights_path.exists():
        print(f"ERROR: {weights_path} not found")
        return

    snap = torch.load(snap_path, map_location="cpu", weights_only=True)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)

    print(f"paft_snapshot.pt keys: {list(snap.keys())}")
    print(f"adapted_weights.pt keys: {list(weights.keys())}")

    for proj in ["V", "O"]:
        check_projection(proj, snap, weights, args.layer, args.head)


if __name__ == "__main__":
    main()
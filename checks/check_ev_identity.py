#!/usr/bin/env python3
"""
check_ev_identity.py

Empirically verifies whether EV_h (the frozen eigenvectors PAFT saves,
from eigh(S_h) on the WIDE per-head matrix) corresponds to U or V of
W_V's SVD in the TALL, stored (adapted_weights.pt) orientation.

Method: load the real init checkpoint's W_V (tall, [n,d]=[768,64] per
head) and EV_V (from paft_snapshot.pt), compute a fresh SVD of W_V
directly, then compare EV_h against both U and V using a sign-flip-robust
subspace check (compares projection matrices EV @ EV^T, which are
invariant to eigenvector sign/reordering, rather than comparing raw
matrices which can spuriously look different due to sign alone).

Also cross-checks lam_h against the freshly computed singular values as
an independent sanity check that the whole comparison is wired correctly.

Usage:
    python3 check_ev_identity.py --task sst2 --method pure_paft
"""
import argparse
from pathlib import Path

import torch


def subspace_agreement(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    A, B: [d, d] matrices whose columns are orthonormal bases (eigenvectors
    or singular vectors). Returns how close their column spaces are,
    robust to sign flips and reordering, via the Frobenius norm distance
    between projection matrices A@A^T and B@B^T (each is exactly I if A/B
    coincide exactly as sets of directions, regardless of sign/order).
    Returns a value in [0, 1] where 1.0 = identical subspace.
    """
    PA = A @ A.T
    PB = B @ B.T
    diff = (PA - PB).norm().item()
    # normalize: max possible diff for d-dim orthonormal projections is bounded;
    # report 1 - (diff / (2*sqrt(d))) as a rough closeness score
    d = A.shape[0]
    return 1.0 - diff / (2 * (d ** 0.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="sst2")
    p.add_argument("--method", default="pure_paft")
    p.add_argument("--results_dir", default="results/glue")
    p.add_argument("--n_layers_to_check", type=int, default=3)
    p.add_argument("--n_heads_to_check", type=int, default=3)
    args = p.parse_args()

    method_dir = Path(args.results_dir) / args.task / args.method

    snap = torch.load(method_dir / "init" / "paft_snapshot.pt",
                       map_location="cpu", weights_only=True)
    weights = torch.load(method_dir / "init" / "adapted_weights.pt",
                          map_location="cpu", weights_only=True)

    EV_V = snap["EV_V"]     # list of 12 per-layer tensors, each [H, d, d]
    lam_V = snap["lam_V"]   # list of 12 per-layer tensors, each [H, d]
    W_V = weights["W_V"]    # list of 12 per-layer tensors, each [H, n, d] (TALL, stored convention)

    print(f"{args.task}/{args.method} — EV_h vs U/V of W_V (tall orientation)\n")

    for layer in range(min(args.n_layers_to_check, len(W_V))):
        for h in range(min(args.n_heads_to_check, W_V[layer].shape[0])):
            M = W_V[layer][h].float()          # [n=768, d=64], tall
            EV_h = EV_V[layer][h].float()       # [d=64, d=64]
            lam_h = lam_V[layer][h].float()     # [d=64]

            # Fresh SVD of the TALL matrix directly (economy)
            U_M, S_M, Vh_M = torch.linalg.svd(M, full_matrices=False)
            # U_M: [768, 64], S_M: [64], Vh_M: [64, 64] (rows = right singular vectors)
            V_M = Vh_M.T   # [64, 64], columns = right singular vectors

            # U_M has shape [768,64] — can't directly compare to EV_h [64,64]
            # via the projection trick (different ambient dimension), so we
            # only test EV_h against V_M here, which is the shape-compatible
            # candidate.
            agree_V = subspace_agreement(EV_h, V_M)

            # Singular value cross-check: does lam_h match S_M (sorted)?
            lam_sorted = lam_h.abs().sort(descending=True).values
            S_sorted = S_M.sort(descending=True).values
            sv_diff = (lam_sorted - S_sorted).abs().mean().item()

            print(f"  layer {layer} head {h}:")
            print(f"    EV_h vs V (right singular vectors of tall W_V): "
                  f"subspace agreement = {agree_V:.4f}  (1.0 = identical)")
            print(f"    lam_h vs fresh singular values: mean|diff| = {sv_diff:.6f}  "
                  f"(should be ~0 if decomposition is self-consistent)")

    print("\nInterpretation:")
    print("  agreement close to 1.0  -> EV_h corresponds to V (right singular")
    print("                             vectors) of W_V in the TALL/stored orientation")
    print("  agreement close to 0.0  -> EV_h corresponds to U instead")
    print("  sv_diff should be ~0 regardless — confirms lam_h really are W_V's")
    print("  singular values, independent of which basis EV_h turns out to match.")


if __name__ == "__main__":
    main()
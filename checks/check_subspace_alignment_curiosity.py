#!/usr/bin/env python3
"""
check_subspace_alignment_curiosity.py

CURIOSITY-ONLY DIAGNOSTIC — not intended as paper evidence. This measures
whether two tasks' W_V update matrices (Delta W = W_final - W_init) occupy
similar subspaces, for methods with a genuinely free/unconstrained
parameterization (LoRA, PoLAR). It does NOT measure or claim anything
about downstream interference/merge quality — that would require actually
merging and evaluating (blocked; see conversation).

Note on SVF/PAFT: both freeze a per-weight basis (Q0 for PAFT, U/V from
SVD for SVF) that's shared across all tasks by construction, since every
task starts from the same pretrained checkpoint. That means subspace
alignment for these methods is TAUTOLOGICALLY ~1.0 (or exactly 1.0 for
PAFT, already proven via |Delta Q|_F=0.00e+00 in table_q_drift.tex) —
this script can include them for a sanity-check comparison point, but
re-measuring this is not new evidence, just confirmation.

Method: for each method + task pair, take Delta W_V per head, extract the
column space (via SVD, keeping singular vectors above a noise threshold),
and compute principal angles between the two tasks' column spaces via
scipy.linalg.subspace_angles. Report mean cosine(principal angle) per
method per task pair, averaged across layers/heads — 1.0 = identical
subspace, 0.0 = orthogonal.

Usage:
    python3 check_subspace_alignment_curiosity.py
"""
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import subspace_angles

RESULTS_DIR = Path("../results/glue")

# Keep it small — a handful of task pairs, matching the earlier proposed
# small scope (mix of a large task and small tasks).
TASK_PAIRS = [
    ("sst2", "mrpc"),
    ("rte", "stsb"),
]

# Full roster. BitFit is expected to hit the degenerate zero-matrix case
# (it never touches W_V — confirmed elsewhere this conversation) and will
# print a warning rather than a meaningful score; included anyway so
# that's visible directly rather than silently pre-filtered out.
METHODS = ["lora_r8", "lora_r64", "polar_r8", "svf",
           "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
           "bitfit", "full_ft"]

SINGULAR_VALUE_THRESHOLD_RATIO = 0.05  # keep singular vectors with
                                        # sigma > 5% of the top singular value


def get_delta_w_v(task: str, method: str):
    """Returns list of per-(layer,head) Delta W_V matrices [n, d].

    LoRA needs the offline-merged final weight (adapted_weights_merged.pt,
    produced by recover_lora_weight.py) — the plain adapted_weights.pt at
    'final' is not guaranteed to reflect the merged trained state. This
    matches build_cache.py's own convention exactly (is_lora -> use
    '_merged.pt' suffix for the final tag only; init never needs merging,
    since no adapter exists yet before training starts).
    """
    method_dir = RESULTS_DIR / task / method
    is_lora = "lora" in method
    final_suffix = "_merged.pt" if is_lora else ".pt"

    init_p = method_dir / "init" / "adapted_weights.pt"
    final_p = method_dir / "final" / f"adapted_weights{final_suffix}"
    if not init_p.exists() or not final_p.exists():
        return None

    w_i = torch.load(init_p, map_location="cpu", weights_only=True)
    w_f = torch.load(final_p, map_location="cpu", weights_only=True)

    deltas = []
    for layer in range(len(w_i["W_V"])):
        W_init = w_i["W_V"][layer].float()   # [H, n, d]
        W_final = w_f["W_V"][layer].float()
        for h in range(W_init.shape[0]):
            deltas.append((W_final[h] - W_init[h]).numpy())
    return deltas


def column_space(M: np.ndarray):
    """Return (basis, is_degenerate). is_degenerate=True means M is
    numerically zero — caller aggregates this rather than warning per call."""
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    if S[0] < 1e-10:
        return U[:, :1], True
    keep = S > (S[0] * SINGULAR_VALUE_THRESHOLD_RATIO)
    keep[0] = True  # always keep at least the top direction
    return U[:, keep], False


def mean_alignment(deltas_a, deltas_b):
    """Mean cos(principal angle) across all (layer,head) pairs.
    Returns (score_or_None, n_degenerate, n_total)."""
    scores = []
    n_degenerate = 0
    n_total = len(deltas_a)
    for A, B in zip(deltas_a, deltas_b):
        basis_a, deg_a = column_space(A)
        basis_b, deg_b = column_space(B)
        if deg_a or deg_b:
            n_degenerate += 1
            continue
        angles = subspace_angles(basis_a, basis_b)
        scores.append(np.mean(np.cos(angles)))

    if n_degenerate > n_total * 0.5:
        # Majority of heads are structurally untouched (e.g. BitFit never
        # trains W_V) — a numeric score here would compare noise to noise.
        return None, n_degenerate, n_total
    return (float(np.mean(scores)) if scores else float("nan")), n_degenerate, n_total


def main():
    print("Subspace alignment between task pairs' Delta W_V — CURIOSITY ONLY,")
    print("not paper evidence. 1.0 = identical subspace, 0.0 = orthogonal.\n")

    for task_a, task_b in TASK_PAIRS:
        print(f"{'='*60}\nTask pair: {task_a} vs {task_b}\n{'='*60}")
        for method in METHODS:
            deltas_a = get_delta_w_v(task_a, method)
            deltas_b = get_delta_w_v(task_b, method)
            if deltas_a is None or deltas_b is None:
                print(f"  {method:<20} missing data for one or both tasks, skipped")
                continue
            if len(deltas_a) != len(deltas_b):
                print(f"  {method:<20} layer/head count mismatch, skipped")
                continue
            score, n_deg, n_total = mean_alignment(deltas_a, deltas_b)
            if score is None:
                print(f"  {method:<20} N/A — {n_deg}/{n_total} heads have "
                      f"structurally untouched W_V (near-zero Delta W); "
                      f"a numeric score here would compare noise to noise")
            elif n_deg > 0:
                print(f"  {method:<20} mean alignment = {score:.4f}  "
                      f"({n_deg}/{n_total} heads excluded as near-zero)")
            else:
                print(f"  {method:<20} mean alignment = {score:.4f}")
        print()

    print("Reminder: PAFT/SVF's high alignment here is expected by construction")
    print("(frozen shared basis), not new evidence. Only LoRA/PoLAR's numbers")
    print("are genuinely new, empirical information.")


if __name__ == "__main__":
    main()
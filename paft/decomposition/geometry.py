"""
Geometric measurements for PAFT analysis.
All functions operate on detached tensors and return Python scalars or tensors.
These are used both during training (for logging) and post-training (for Analysis 2-4).
"""

from typing import Optional, Dict
import torch


def rotation_drift(Q: torch.Tensor, Q_ref: torch.Tensor) -> float:
    """
    ||Q - Q_ref||_F  — how much Q has moved from a reference (e.g. initial frozen state).
    For pure/hybrid PAFT, Q is always frozen so this should be ~0.
    Useful for validating that the freeze constraint holds.
    """
    return torch.norm(Q - Q_ref, p='fro').item()


def scaling_change(S_init: torch.Tensor, S_final: torch.Tensor) -> float:
    """||S_final - S_init||_F"""
    return torch.norm(S_final - S_init, p='fro').item()


def geometry_ratio(drift: float, s_change: float, eps: float = 1e-10) -> float:
    """
    drift / (drift + scaling_change).
    0 = pure scaling adaptation, 1 = pure rotation adaptation.
    For all PAFT variants, Q is frozen so this should be ~0.
    Useful to detect numerical drift (e.g. from optimizer eps).
    """
    return drift / (drift + s_change + eps)


def effective_rank(S: torch.Tensor, threshold: float = 0.01) -> int:
    """
    Number of eigenvalues above threshold * max_eigenvalue.
    Measures how many scaling directions S is actually using.
    """
    eigs = torch.linalg.eigvalsh(S).abs()
    max_eig = eigs.max().item()
    if max_eig < 1e-10:
        return 0
    return int((eigs > threshold * max_eig).sum().item())


def eigenvalue_shift(S_init: torch.Tensor, S_final: torch.Tensor) -> torch.Tensor:
    """
    Per-eigenvector shift: projects S_final onto S_init's eigenbasis and returns
    lambda_final_in_init_basis - lambda_init.
    Shape: [d_head]
    This is the key interpretability signal — large shifts in a direction mean
    that direction was important for domain adaptation.
    """
    lam_init, V_init = torch.linalg.eigh(S_init)
    # Project S_final onto init eigenbasis: lambda_j = v_j^T S_final v_j
    lam_final_projected = torch.diag(V_init.mT @ S_final @ V_init)
    return lam_final_projected - lam_init


def compute_head_metrics(
    Q_V_init: torch.Tensor,
    Q_O_init: torch.Tensor,
    S_V_init: torch.Tensor,
    S_O_init: torch.Tensor,
    S_V_final: torch.Tensor,
    S_O_final: torch.Tensor,
    Q_V_final: Optional[torch.Tensor] = None,
    Q_O_final: Optional[torch.Tensor] = None,
) -> Dict:
    """
    All geometric metrics for a single (layer, head) pair.
    Q_V_final / Q_O_final only needed for methods that unfreeze Q (none currently,
    but included for completeness and forward-compatibility with hybrid-Q variants).

    Returns dict with all scalar and tensor metrics.
    """
    # Rotation drift — should be ~0 for all PAFT variants (Q is frozen)
    drift_V = rotation_drift(Q_V_final, Q_V_init) if Q_V_final is not None else 0.0
    drift_O = rotation_drift(Q_O_final, Q_O_init) if Q_O_final is not None else 0.0

    s_change_V = scaling_change(S_V_init, S_V_final)
    s_change_O = scaling_change(S_O_init, S_O_final)

    return {
        "rotation_drift_V":      drift_V,
        "rotation_drift_O":      drift_O,
        "scaling_change_V":      s_change_V,
        "scaling_change_O":      s_change_O,
        "geometry_ratio_V":      geometry_ratio(drift_V, s_change_V),
        "geometry_ratio_O":      geometry_ratio(drift_O, s_change_O),
        "effective_rank_V":      effective_rank(S_V_final),
        "effective_rank_O":      effective_rank(S_O_final),
        "delta_lambda_V":        eigenvalue_shift(S_V_init, S_V_final),
        "delta_lambda_O":        eigenvalue_shift(S_O_init, S_O_final),
    }


def compute_model_metrics(
    init_tensors: Dict,
    final_tensors: Dict,
    n_layers: int,
    n_heads: int,
) -> Dict:
    """
    Compute head metrics for all (layer, head) pairs.
    init_tensors / final_tensors: dicts with keys 'S_V', 'S_O', 'Q_V', 'Q_O'
      each a list of length n_layers, each element [n_heads, d_head, d_head] or [n_heads, d_model, d_head].

    Returns nested dict: results[layer_idx][head_idx] = metric_dict.
    """
    results = {}
    for l in range(n_layers):
        results[l] = {}
        S_V_init_l = init_tensors['S_V'][l]     # [n_heads, d_head, d_head]
        S_O_init_l = init_tensors['S_O'][l]
        S_V_final_l = final_tensors['S_V'][l]
        S_O_final_l = final_tensors['S_O'][l]
        Q_V_init_l = init_tensors['Q_V'][l]
        Q_O_init_l = init_tensors['Q_O'][l]

        for h in range(n_heads):
            results[l][h] = compute_head_metrics(
                Q_V_init=Q_V_init_l[h],
                Q_O_init=Q_O_init_l[h],
                S_V_init=S_V_init_l[h],
                S_O_init=S_O_init_l[h],
                S_V_final=S_V_final_l[h],
                S_O_final=S_O_final_l[h],
            )
    return results
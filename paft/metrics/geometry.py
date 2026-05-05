from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Any


def compute_spectrum_metrics(W: torch.Tensor) -> Dict[str, float]:
    """
    Computes the spectral fingerprint of a weight matrix.
    Operates on [n_embd, d_head] or [n_embd, n_embd].
    """
    # Use standard SVD for the audit; move to CPU to protect 8GB VRAM
    sigma = torch.linalg.svdvals(W.detach().cpu().float())
    sigma_sq = sigma ** 2
    total = sigma_sq.sum()
    p = sigma_sq / (total + 1e-10)

    # AI Physicist Metrics
    return {
        "stable_rank": (total / (sigma.max() ** 2 + 1e-10)).item(),
        "sv_entropy": -(p * (p + 1e-10).log()).sum().item(),
        "eff_rank": (-(p * (p + 1e-10).log()).sum().exp()).item(),
        "condition_num": (sigma.max() / (sigma.min() + 1e-10)).item(),
        "isotropy": (sigma.min() / (sigma.max() + 1e-10)).item(),
    }


def geometric_preservation_score(metrics_pre: Dict, metrics_post: Dict) -> float:
    """Calculates the 0.0 to 1.0 preservation score."""
    deltas = []
    for k in metrics_pre:
        d = (metrics_post[k] - metrics_pre[k]) / (abs(metrics_pre[k]) + 1e-10)
        deltas.append(abs(d))
    return 1.0 - np.mean(deltas)
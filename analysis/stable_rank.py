"""
Stable rank analysis for PAFT experiments.

This is the primary novel geometric analysis:
  1. Stable rank of ΔW (replicates PoLAR's analysis)
  2. Stable rank of W_effective = W_0 + ΔW (the novel contribution — neither
     LoRA nor PoLAR computes this)
  3. Directional diversity of W_effective (pairwise distance of normalised rows)
  4. Rotation drift — how much Q drifts during training (should be ~0 for PAFT)
  5. Spectral entropy of W_effective (measures concentration of singular values)

Key result to show:
  PoLAR improves sr(ΔW) significantly vs LoRA (their main result).
  PAFT improves sr(W_eff) — the stable rank of the FULL effective weight matrix.
  These are different objects: sr(ΔW) doesn't imply sr(W_eff) is improved.
  Proposition 1 in the paper: PoLAR cannot guarantee sr(W_eff) because
  the Stiefel constraint applies to ΔW, not to W_0 + ΔW.

Definitions:
  Stable rank: sr(W) = ||W||_F² / ||W||_2²  ∈ [1, min(m,n)]
    Frobenius norm squared / largest singular value squared.
    Equals the effective number of "active" singular dimensions.
    High sr = geometrically diverse weight matrix.
    Low sr = weight matrix dominated by a few directions (collapsed).

  Spectral entropy: H(W) = -Σ_i p_i log(p_i)  where p_i = σ_i² / Σ σ_j²
    Entropy of the normalised singular value distribution.
    Maximum when all σ are equal (uniform), minimum when one dominates.

  Effective rank: er(W) = exp(H(W))
    Exponentiated spectral entropy — scale-free version of stable rank.
    Roy & Vetterli 2007.

Usage:
    from paft.analysis.stable_rank import (
        stable_rank, spectral_entropy, effective_rank,
        analyze_method_weights, compare_methods_stable_rank,
    )

    # After training, load saved adapted weights and compute metrics
    results = analyze_method_weights(W_eff_per_layer, W_init_per_layer)
    plot_stable_rank_comparison(results_dict, save_path="figures/stable_rank.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core scalar metrics
# ──────────────────────────────────────────────────────────────────────────────

def stable_rank(W: torch.Tensor) -> float:
    """
    Stable rank: sr(W) = ||W||_F² / ||W||_2²

    Properties:
      - Always in [1, min(m,n)]
      - Numerically stable (unlike matrix rank, no threshold needed)
      - sr = 1 → W has rank 1 (completely collapsed)
      - sr = min(m,n) → all singular values equal (maximally diverse)

    Args:
        W: 2D tensor of any shape [m, n].

    Returns:
        Scalar float.
    """
    if W.ndim != 2:
        raise ValueError(f"stable_rank expects 2D tensor, got shape {W.shape}")
    W_f = W.detach().float()
    frobenius_sq = (W_f ** 2).sum().item()
    spectral_sq  = torch.linalg.svdvals(W_f)[0].item() ** 2
    if spectral_sq < 1e-12:
        return 1.0
    return frobenius_sq / spectral_sq


def spectral_entropy(W: torch.Tensor) -> float:
    """
    Spectral entropy: H(W) = -Σ_i (σ_i² / Σ σ_j²) log(σ_i² / Σ σ_j²)

    Returns:
        Entropy in nats.  Range: [0, log(min(m,n))].
    """
    W_f = W.detach().float()
    sigma = torch.linalg.svdvals(W_f)
    p = sigma ** 2
    p = p / p.sum().clamp(min=1e-12)
    p = p[p > 1e-12]   # avoid log(0)
    return (-p * p.log()).sum().item()


def effective_rank(W: torch.Tensor) -> float:
    """
    Effective rank: er(W) = exp(H(W)) — exponentiated spectral entropy.
    Roy & Vetterli 2007 definition.  More principled than stable rank but
    equivalent in practice for comparing methods.
    """
    return float(np.exp(spectral_entropy(W)))


def condition_number(W: torch.Tensor) -> float:
    """
    σ_max / σ_min.  Numerically related to invertibility.
    High condition number = matrix close to singular, dominanted by top directions.
    """
    W_f = W.detach().float()
    sv = torch.linalg.svdvals(W_f)
    if sv[-1].item() < 1e-12:
        return float('inf')
    return (sv[0] / sv[-1]).item()


def directional_diversity(W: torch.Tensor) -> float:
    """
    Mean pairwise Euclidean distance of unit-normalised rows of W.
    High = rows point in diverse directions (like PoLAR's Figure 3).
    Low = rows cluster into a few directions (LoRA collapse phenomenon).

    Replicates PoLAR's directional diversity analysis.
    """
    W_f = W.detach().float()
    norms = W_f.norm(dim=1, keepdim=True).clamp(min=1e-12)
    W_norm = W_f / norms   # unit-normalised rows [m, n]
    # Pairwise distances — O(m²n) but only called post-hoc on small matrices
    dists = torch.cdist(W_norm, W_norm, p=2)  # [m, m]
    m = dists.shape[0]
    if m <= 1:
        return 0.0
    # Upper triangle (no diagonal)
    mask = torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)
    return dists[mask].mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Layer-level analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_weight_matrix(
    W_eff:  torch.Tensor,
    W_init: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Full geometric health metrics for one weight matrix.

    Args:
        W_eff:  Effective weight after fine-tuning [m, n].
        W_init: Initial pretrained weight [m, n].  If provided, also computes
                ΔW metrics (sr(ΔW), sr(W_eff) vs sr(ΔW) gap).
    """
    metrics = {
        "stable_rank_Weff":       stable_rank(W_eff),
        "spectral_entropy_Weff":  spectral_entropy(W_eff),
        "effective_rank_Weff":    effective_rank(W_eff),
        "condition_number_Weff":  condition_number(W_eff),
        "directional_diversity":  directional_diversity(W_eff),
    }

    if W_init is not None:
        delta_W = W_eff - W_init
        sr_delta = stable_rank(delta_W)
        metrics["stable_rank_delta_W"]       = sr_delta
        metrics["stable_rank_W_init"]        = stable_rank(W_init)
        metrics["sr_Weff_minus_sr_deltaW"]   = metrics["stable_rank_Weff"] - sr_delta
        metrics["frobenius_norm_delta_W"]    = delta_W.norm().item()
        metrics["spectral_norm_delta_W"]     = torch.linalg.svdvals(delta_W.float())[0].item()

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Model-level analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_all_layers(
    W_eff_per_layer:  List[torch.Tensor],
    W_init_per_layer: Optional[List[torch.Tensor]] = None,
) -> Dict[str, List[float]]:
    """
    Compute stable rank and related metrics for all layers.

    Args:
        W_eff_per_layer:  List of [m, n] effective weight tensors (one per layer).
        W_init_per_layer: Optional list of [m, n] initial weight tensors.

    Returns:
        Dict mapping metric_name → list of scalar values (one per layer).
        Convenient for plotting layer-by-layer profiles.
    """
    n_layers = len(W_eff_per_layer)
    has_init = W_init_per_layer is not None

    # Collect metrics per layer
    all_metrics: Dict[str, List[float]] = {}

    for l in range(n_layers):
        W_eff  = W_eff_per_layer[l]
        W_init = W_init_per_layer[l] if has_init else None

        m = analyze_weight_matrix(W_eff, W_init)
        for k, v in m.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    return all_metrics


def summarize_stable_rank(
    W_eff_per_layer: List[torch.Tensor],
    W_init_per_layer: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Mean stable rank metrics aggregated across all layers.
    Produces the scalar numbers reported in the main results table.
    """
    layer_metrics = analyze_all_layers(W_eff_per_layer, W_init_per_layer)
    return {
        k: float(np.mean(v)) for k, v in layer_metrics.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-method comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_methods_stable_rank(
    methods_W_eff: Dict[str, List[torch.Tensor]],
    W_init_per_layer: Optional[List[torch.Tensor]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare stable rank metrics across multiple methods.

    Args:
        methods_W_eff:    {method_name: [W_eff per layer]} for each method.
        W_init_per_layer: Shared pretrained weights for ΔW computation.

    Returns:
        Nested dict: method_name → {metric_name: mean_value}
        Ready to be converted to a pandas DataFrame for tables.

    Example output:
        {
            "paft_hybrid": {"stable_rank_Weff": 12.3, "stable_rank_delta_W": 4.1, ...},
            "lora_r8":     {"stable_rank_Weff":  8.1, "stable_rank_delta_W": 1.06, ...},
            "polar":       {"stable_rank_Weff":  9.2, "stable_rank_delta_W": 5.2, ...},
        }

    This directly produces the data for Analysis 2 in the paper:
    "Show that PoLAR's Stiefel constraint improves sr(ΔW) but not sr(W_0 + ΔW)."
    """
    results = {}
    for method_name, W_eff_layers in methods_W_eff.items():
        results[method_name] = summarize_stable_rank(W_eff_layers, W_init_per_layer)
        logger.info(
            f"{method_name}: sr(W_eff)={results[method_name].get('stable_rank_Weff', 'N/A'):.2f}  "
            f"sr(ΔW)={results[method_name].get('stable_rank_delta_W', 'N/A'):.2f}"
        )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Stable rank dynamics during training (PoLAR Analysis 4 equivalent)
# ──────────────────────────────────────────────────────────────────────────────

def track_stable_rank_over_steps(
    W_snapshots: List[List[torch.Tensor]],   # [step][layer] → W_eff [m, n]
    steps: List[int],
) -> Dict[str, List[float]]:
    """
    Track stable rank evolution over training steps.

    Args:
        W_snapshots: Ordered list of per-step weight snapshots.
                     Each element is a list of [m, n] tensors (one per layer).
        steps:       Training step indices (for x-axis of plot).

    Returns:
        {"step": steps, "mean_sr_Weff": [...], "std_sr_Weff": [...]}
    """
    mean_srs, std_srs = [], []

    for W_layers in W_snapshots:
        srs = [stable_rank(W) for W in W_layers]
        mean_srs.append(float(np.mean(srs)))
        std_srs.append(float(np.std(srs)))

    return {
        "step":       steps,
        "mean_sr_Weff": mean_srs,
        "std_sr_Weff":  std_srs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting utilities (requires matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def plot_stable_rank_comparison(
    comparison: Dict[str, Dict[str, float]],
    metric: str = "stable_rank_Weff",
    save_path: Optional[str] = None,
    title: str = "Stable Rank of W_effective by Method",
) -> None:
    """
    Bar chart comparing stable rank across methods.
    Replicates PoLAR's Figure 4 style but for sr(W_eff) instead of sr(ΔW).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    methods = list(comparison.keys())
    values  = [comparison[m].get(metric, 0.0) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2196F3' if 'paft' in m.lower() else '#9E9E9E' for m in methods]
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha='center', va='bottom', fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_layer_profile(
    layer_metrics: Dict[str, List[float]],
    metric: str = "stable_rank_Weff",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Line plot of stable rank per layer — shows which layers adapt most.
    Replicates PoLAR's stable rank dynamics analysis (their Figure 2).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    values = layer_metrics.get(metric, [])
    if not values:
        logger.warning(f"Metric '{metric}' not found in layer_metrics")
        return

    layers = list(range(len(values)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, values, 'o-', color='#2196F3', linewidth=1.5, markersize=4)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or metric.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Quantization noise floor measurement
# ──────────────────────────────────────────────────────────────────────────────

def measure_quantization_noise(
    W_fp32: torch.Tensor,
    W_quantized: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare geometric health metrics between fp32 and quantized weight matrices.
    Used to verify that NF4 quantization does not materially affect stable rank.

    Call this at init time before any training:
        noise = measure_quantization_noise(W_original_fp32, W_after_nf4_dequant)

    The paper should report: "Quantization reduces sr(W_V) by {delta:.2f} on average,
    which is {ratio:.1%} of the pretrained value — negligible compared to the
    sr changes induced by fine-tuning."
    """
    sr_fp32  = stable_rank(W_fp32)
    sr_q     = stable_rank(W_quantized)
    delta    = W_fp32 - W_quantized
    return {
        "sr_fp32":           sr_fp32,
        "sr_quantized":      sr_q,
        "sr_delta":          sr_fp32 - sr_q,
        "sr_relative_error": abs(sr_fp32 - sr_q) / max(sr_fp32, 1e-6),
        "frobenius_error":   delta.norm().item(),
        "relative_frob_error": (delta.norm() / W_fp32.norm().clamp(1e-6)).item(),
    }
"""
Tests for paft/decomposition/geometry.py

Verifies that geometric metrics are mathematically correct using matrices
with known properties.  No neural network loading required.
"""

import math
import pytest
import torch

from paft.decomposition.geometry import (
    rotation_drift,
    scaling_change,
    geometry_ratio,
    effective_rank,
    eigenvalue_shift,
    compute_head_metrics,
)


# ──────────────────────────────────────────────────────────────────────────────
# rotation_drift
# ──────────────────────────────────────────────────────────────────────────────

def test_rotation_drift_zero_for_identical():
    Q = torch.randn(768, 64)
    assert rotation_drift(Q, Q.clone()) == pytest.approx(0.0, abs=1e-6)


def test_rotation_drift_positive_for_different():
    Q1 = torch.randn(768, 64)
    Q2 = torch.randn(768, 64)
    assert rotation_drift(Q1, Q2) > 0.0


def test_rotation_drift_is_frobenius_norm():
    torch.manual_seed(0)
    Q1 = torch.randn(64, 64)
    Q2 = torch.randn(64, 64)
    expected = torch.norm(Q1 - Q2, p='fro').item()
    assert rotation_drift(Q1, Q2) == pytest.approx(expected, rel=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# scaling_change
# ──────────────────────────────────────────────────────────────────────────────

def test_scaling_change_zero_for_identical():
    S = torch.randn(64, 64)
    assert scaling_change(S, S.clone()) == pytest.approx(0.0, abs=1e-6)


def test_scaling_change_positive_for_different():
    S1 = torch.randn(64, 64)
    S2 = torch.randn(64, 64)
    assert scaling_change(S1, S2) > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# geometry_ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_geometry_ratio_zero_when_no_rotation():
    """Pure scaling adaptation — ratio should be 0"""
    assert geometry_ratio(drift=0.0, s_change=1.0) == pytest.approx(0.0, abs=1e-6)


def test_geometry_ratio_near_one_when_only_rotation():
    """Almost pure rotation — ratio should approach 1"""
    ratio = geometry_ratio(drift=1000.0, s_change=0.001)
    assert ratio > 0.999


def test_geometry_ratio_half_for_equal():
    """Equal drift and scaling change — ratio should be 0.5"""
    ratio = geometry_ratio(drift=1.0, s_change=1.0)
    assert ratio == pytest.approx(0.5, abs=1e-4)


def test_geometry_ratio_no_divide_by_zero():
    """Both zero — eps prevents division by zero"""
    ratio = geometry_ratio(drift=0.0, s_change=0.0)
    assert 0.0 <= ratio <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# effective_rank
# ──────────────────────────────────────────────────────────────────────────────

def test_effective_rank_identity_is_full_rank():
    """Identity matrix has all eigenvalues = 1 — effective rank = d"""
    I = torch.eye(64)
    assert effective_rank(I) == 64


def test_effective_rank_rank1_matrix():
    """Rank-1 PSD matrix has effective rank 1"""
    v = torch.randn(64)
    S = torch.outer(v, v)
    assert effective_rank(S) == 1


def test_effective_rank_zero_matrix():
    """All-zero matrix has effective rank 0"""
    assert effective_rank(torch.zeros(64, 64)) == 0


def test_effective_rank_bounded_by_dim():
    S = torch.randn(32, 32)
    S = S @ S.T   # PSD
    assert 0 <= effective_rank(S) <= 32


# ──────────────────────────────────────────────────────────────────────────────
# eigenvalue_shift
# ──────────────────────────────────────────────────────────────────────────────

def test_eigenvalue_shift_zero_for_identical():
    """No change in S → zero eigenvalue shift everywhere"""
    A = torch.randn(64, 64)
    S = A @ A.T
    shift = eigenvalue_shift(S, S.clone())
    assert shift.abs().max() < 1e-3


def test_eigenvalue_shift_detects_scaling():
    """Scaling S by 2 should shift each eigenvalue by +lambda"""
    A = torch.randn(64, 64)
    S = A @ A.T
    S2 = 2.0 * S
    shift = eigenvalue_shift(S, S2)
    # Expected: shift ≈ lambda_i (since 2*lambda - lambda = lambda)
    lam, _ = torch.linalg.eigh(S)
    assert torch.allclose(shift.sort().values, lam.sort().values, atol=1e-3), (
        "Doubling S should shift eigenvalues by +lambda_i"
    )


def test_eigenvalue_shift_shape():
    S = torch.eye(64)
    shift = eigenvalue_shift(S, S + 0.1 * torch.eye(64))
    assert shift.shape == (64,)


# ──────────────────────────────────────────────────────────────────────────────
# compute_head_metrics
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_head_metrics_keys():
    """Returns all expected keys"""
    torch.manual_seed(0)
    Q_V = torch.randn(768, 64);  Q_O = torch.randn(64, 768)
    A = torch.randn(64, 64);     S = A @ A.T
    metrics = compute_head_metrics(Q_V, Q_O, S, S, S, S)

    expected_keys = {
        "rotation_drift_V", "rotation_drift_O",
        "scaling_change_V", "scaling_change_O",
        "geometry_ratio_V", "geometry_ratio_O",
        "effective_rank_V", "effective_rank_O",
        "delta_lambda_V",   "delta_lambda_O",
    }
    assert expected_keys == set(metrics.keys())


def test_compute_head_metrics_no_change_is_zero():
    """If S_init == S_final, all change metrics are zero"""
    torch.manual_seed(0)
    Q_V = torch.randn(768, 64);  Q_O = torch.randn(64, 768)
    A = torch.randn(64, 64);     S = A @ A.T

    metrics = compute_head_metrics(Q_V, Q_O, S, S, S, S)

    assert metrics["scaling_change_V"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["scaling_change_O"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["geometry_ratio_V"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["delta_lambda_V"].abs().max() < 1e-4


def test_compute_head_metrics_rotation_drift_with_explicit_Q():
    """Passing different Q_final and Q_init produces non-zero rotation drift"""
    torch.manual_seed(0)
    Q_V_init = torch.randn(768, 64)
    Q_V_final = torch.randn(768, 64)
    Q_O = torch.randn(64, 768)
    A = torch.randn(64, 64);  S = A @ A.T

    metrics = compute_head_metrics(
        Q_V_init, Q_O, S, S, S, S,
        Q_V_final=Q_V_final, Q_O_final=Q_O
    )
    assert metrics["rotation_drift_V"] > 0.0
    assert metrics["rotation_drift_O"] == pytest.approx(0.0, abs=1e-6)
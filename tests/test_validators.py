"""
Tests for paft/decomposition/validators.py

Verifies that each validator correctly accepts valid inputs and raises
AssertionError with informative messages for invalid inputs.
"""

import pytest
import torch

from paft.decomposition.validators import (
    assert_orthogonal_columns,
    assert_orthogonal_rows,
    assert_symmetric,
    assert_psd,
    assert_symmetric_psd,
    assert_reconstruction,
)


# ──────────────────────────────────────────────────────────────────────────────
# assert_orthogonal_columns
# ──────────────────────────────────────────────────────────────────────────────

def test_orthogonal_columns_accepts_valid():
    Q, _ = torch.linalg.qr(torch.randn(768, 64))
    assert_orthogonal_columns(Q)  # should not raise


def test_orthogonal_columns_rejects_non_orthogonal():
    Q = torch.randn(768, 64)  # random matrix, not orthogonal
    with pytest.raises(AssertionError, match="orthogonality"):
        assert_orthogonal_columns(Q, atol=1e-4)


def test_orthogonal_columns_tight_tolerance():
    """Near-orthogonal matrix passes loose tolerance, fails tight"""
    Q, _ = torch.linalg.qr(torch.randn(768, 64))
    Q_noisy = Q + 1e-3 * torch.randn_like(Q)
    assert_orthogonal_columns(Q_noisy, atol=1e-2)  # passes
    with pytest.raises(AssertionError):
        assert_orthogonal_columns(Q_noisy, atol=1e-5)  # fails


def test_orthogonal_columns_requires_2d():
    with pytest.raises(AssertionError, match="2-D"):
        assert_orthogonal_columns(torch.randn(4, 4, 4))


# ──────────────────────────────────────────────────────────────────────────────
# assert_orthogonal_rows
# ──────────────────────────────────────────────────────────────────────────────

def test_orthogonal_rows_accepts_valid():
    Q, _ = torch.linalg.qr(torch.randn(768, 64))
    Q_rows = Q.T   # [64, 768] — orthonormal rows
    assert_orthogonal_rows(Q_rows)


def test_orthogonal_rows_rejects_non_orthogonal():
    Q = torch.randn(64, 768)
    with pytest.raises(AssertionError, match="orthogonality"):
        assert_orthogonal_rows(Q, atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# assert_symmetric
# ──────────────────────────────────────────────────────────────────────────────

def test_symmetric_accepts_valid():
    A = torch.randn(64, 64)
    S = A @ A.T   # always symmetric
    assert_symmetric(S)


def test_symmetric_rejects_asymmetric():
    S = torch.randn(64, 64)   # random, almost certainly asymmetric
    with pytest.raises(AssertionError, match="symmetry"):
        assert_symmetric(S, atol=1e-4)


def test_symmetric_requires_square():
    with pytest.raises(AssertionError):
        assert_symmetric(torch.randn(64, 32))


# ──────────────────────────────────────────────────────────────────────────────
# assert_psd
# ──────────────────────────────────────────────────────────────────────────────

def test_psd_accepts_positive_definite():
    A = torch.randn(64, 64)
    S = A @ A.T + 0.1 * torch.eye(64)   # guaranteed PD
    assert_psd(S)


def test_psd_accepts_semidefinite():
    A = torch.randn(32, 64)
    S = A.T @ A   # rank <= 32, PSD
    assert_psd(S)


def test_psd_rejects_negative_definite():
    A = torch.randn(64, 64)
    S = -(A @ A.T + 0.1 * torch.eye(64))   # negative definite
    with pytest.raises(AssertionError, match="PSD"):
        assert_psd(S)


# ──────────────────────────────────────────────────────────────────────────────
# assert_symmetric_psd (combined)
# ──────────────────────────────────────────────────────────────────────────────

def test_symmetric_psd_accepts_polar_S():
    """S from polar decomposition should pass both checks"""
    from paft.decomposition.polar import polar_decompose
    torch.manual_seed(0)
    W = torch.randn(768, 64)
    _, S = polar_decompose(W)
    assert_symmetric_psd(S)


def test_symmetric_psd_rejects_symmetric_but_not_psd():
    A = torch.randn(64, 64)
    S = A @ A.T + 0.1 * torch.eye(64)
    S_neg = -S   # symmetric but negative definite
    with pytest.raises(AssertionError):
        assert_symmetric_psd(S_neg)


# ──────────────────────────────────────────────────────────────────────────────
# assert_reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def test_reconstruction_accepts_exact_match():
    W = torch.randn(768, 64)
    assert_reconstruction(W, W.clone())


def test_reconstruction_accepts_within_tolerance():
    W = torch.randn(768, 64)
    W_noisy = W + 1e-5 * torch.randn_like(W)
    assert_reconstruction(W, W_noisy, atol=1e-3)


def test_reconstruction_rejects_large_error():
    W = torch.randn(768, 64)
    W_wrong = W + 1.0
    with pytest.raises(AssertionError, match="reconstruction"):
        assert_reconstruction(W, W_wrong, atol=1e-4)


def test_reconstruction_custom_label_in_message():
    W = torch.randn(10, 5)
    W_wrong = W + 1.0
    with pytest.raises(AssertionError, match="my_layer"):
        assert_reconstruction(W, W_wrong, atol=1e-4, label="my_layer")
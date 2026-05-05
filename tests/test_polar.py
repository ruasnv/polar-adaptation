"""
Tests for paft/decomposition/polar.py

Tests run with synthetic random matrices — no GPT-2 download required.
Every test is independent and fast (< 1s each on CPU).

Mathematical contracts verified:
  - W = Q @ S  (right polar reconstruction)
  - W = S @ Q  (left polar reconstruction)
  - Q^T Q = I  (column orthonormality for right polar)
  - Q Q^T = I  (row orthonormality for left polar)
  - S symmetric
  - S PSD (eigenvalues >= 0)
  - Numerical stability with ill-conditioned matrices
  - Safe SVD fallback path (mocked LinAlgError)
"""

import pytest
import torch
from unittest.mock import patch

from paft.decomposition.polar import (
    polar_decompose,
    polar_decompose_left,
    safe_svd,
    reconstruct,
    reconstruct_left,
)
from paft.decomposition.validators import (
    assert_orthogonal_columns,
    assert_orthogonal_rows,
    assert_symmetric_psd,
    assert_reconstruction,
)

ATOL = 1e-4


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: weight shapes matching GPT-2 small and medium
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[
    (768, 64),   # GPT-2 small W_V_h shape  [n_embd, d_head]
    (64, 768),   # GPT-2 small W_O_h shape  [d_head, n_embd]
    (1024, 64),  # GPT-2 medium W_V_h
    (64, 1024),  # GPT-2 medium W_O_h
    (64, 64),    # square
])
def random_matrix(request):
    m, n = request.param
    torch.manual_seed(0)
    return torch.randn(m, n, dtype=torch.float32)


@pytest.fixture
def W_V():
    """Tall matrix: GPT-2 small W_V_h [n_embd=768, d_head=64]"""
    torch.manual_seed(42)
    return torch.randn(768, 64, dtype=torch.float32)


@pytest.fixture
def W_O():
    """Wide matrix: GPT-2 small W_O_h [d_head=64, n_embd=768]"""
    torch.manual_seed(42)
    return torch.randn(64, 768, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Right polar decomposition (W_V — tall)
# ──────────────────────────────────────────────────────────────────────────────

def test_right_polar_reconstruction(W_V):
    """W = Q @ S"""
    Q, S = polar_decompose(W_V)
    W_rec = reconstruct(Q, S)
    assert_reconstruction(W_V, W_rec, atol=ATOL, label="W_V right polar")


def test_right_polar_Q_orthonormal_columns(W_V):
    """Q^T Q = I_{d_head}"""
    Q, S = polar_decompose(W_V)
    assert_orthogonal_columns(Q, atol=ATOL, label="Q_V")


def test_right_polar_S_symmetric_psd(W_V):
    """S is symmetric PSD"""
    Q, S = polar_decompose(W_V)
    assert_symmetric_psd(S, atol=ATOL, label="S_V")


def test_right_polar_shapes(W_V):
    """Output shapes match contract"""
    m, n = W_V.shape
    Q, S = polar_decompose(W_V)
    assert Q.shape == (m, n), f"Q shape {Q.shape} != ({m},{n})"
    assert S.shape == (n, n), f"S shape {S.shape} != ({n},{n})"


def test_right_polar_rejects_wide():
    """polar_decompose raises for wide matrices"""
    W = torch.randn(64, 768)
    with pytest.raises(ValueError, match="m >= n"):
        polar_decompose(W)


# ──────────────────────────────────────────────────────────────────────────────
# Left polar decomposition (W_O — wide)
# ──────────────────────────────────────────────────────────────────────────────

def test_left_polar_reconstruction(W_O):
    """W = S @ Q"""
    S, Q = polar_decompose_left(W_O)
    W_rec = reconstruct_left(S, Q)
    assert_reconstruction(W_O, W_rec, atol=ATOL, label="W_O left polar")


def test_left_polar_Q_orthonormal_rows(W_O):
    """Q Q^T = I_{d_head}"""
    S, Q = polar_decompose_left(W_O)
    assert_orthogonal_rows(Q, atol=ATOL, label="Q_O")


def test_left_polar_S_symmetric_psd(W_O):
    """S is symmetric PSD"""
    S, Q = polar_decompose_left(W_O)
    assert_symmetric_psd(S, atol=ATOL, label="S_O")


def test_left_polar_shapes(W_O):
    """Output shapes match contract"""
    m, n = W_O.shape
    S, Q = polar_decompose_left(W_O)
    assert S.shape == (m, m), f"S shape {S.shape} != ({m},{m})"
    assert Q.shape == (m, n), f"Q shape {Q.shape} != ({m},{n})"


def test_left_polar_rejects_tall():
    """polar_decompose_left raises for tall matrices"""
    W = torch.randn(768, 64)
    with pytest.raises(ValueError, match="m <= n"):
        polar_decompose_left(W)


# ──────────────────────────────────────────────────────────────────────────────
# Square matrices
# ──────────────────────────────────────────────────────────────────────────────

def test_square_right_polar():
    torch.manual_seed(7)
    W = torch.randn(64, 64)
    Q, S = polar_decompose(W)
    assert_reconstruction(W, Q @ S, atol=ATOL)
    assert_orthogonal_columns(Q, atol=ATOL)
    assert_symmetric_psd(S, atol=ATOL)


def test_square_left_polar():
    torch.manual_seed(7)
    W = torch.randn(64, 64)
    S, Q = polar_decompose_left(W)
    assert_reconstruction(W, S @ Q, atol=ATOL)
    assert_orthogonal_rows(Q, atol=ATOL)
    assert_symmetric_psd(S, atol=ATOL)


# ──────────────────────────────────────────────────────────────────────────────
# Numerical stability
# ──────────────────────────────────────────────────────────────────────────────

def test_nearly_rank_deficient_matrix():
    """Near-singular matrix still decomposes correctly"""
    torch.manual_seed(0)
    W = torch.randn(768, 64)
    # Make nearly rank-deficient: zero out most singular values
    U, s, Vh = torch.linalg.svd(W, full_matrices=False)
    s_small = s.clone()
    s_small[32:] = 1e-8  # last 32 singular values near-zero
    W_ill = U @ torch.diag(s_small) @ Vh

    Q, S = polar_decompose(W_ill)
    W_rec = Q @ S
    assert_reconstruction(W_ill, W_rec, atol=1e-3)
    assert_orthogonal_columns(Q, atol=1e-3)


def test_safe_svd_fallback():
    """safe_svd recovers via perturbation when SVD raises LinAlgError"""
    torch.manual_seed(0)
    W = torch.randn(64, 64)

    call_count = {"n": 0}
    real_svd = torch.linalg.svd

    def patched_svd(x, full_matrices=True):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise torch.linalg.LinAlgError("forced failure")
        return real_svd(x, full_matrices=full_matrices)

    with patch("torch.linalg.svd", side_effect=patched_svd):
        U, sigma, Vh = safe_svd(W)

    assert call_count["n"] == 2, "fallback path not taken"
    assert U.shape == (64, 64)
    assert sigma.shape == (64,)


def test_identity_matrix():
    """Polar decomposition of identity: Q=I, S=I"""
    I = torch.eye(64)
    Q, S = polar_decompose(I)
    assert (Q - torch.eye(64)).abs().max() < ATOL, "Q should be I for identity input"
    assert (S - torch.eye(64)).abs().max() < ATOL, "S should be I for identity input"


def test_s_eigenvalues_equal_singular_values():
    """Eigenvalues of S equal singular values of W (fundamental property)"""
    torch.manual_seed(0)
    W = torch.randn(768, 64)
    Q, S = polar_decompose(W)

    sv_W  = torch.linalg.svdvals(W).sort(descending=True).values
    ev_S  = torch.linalg.eigvalsh(S).sort(descending=True).values

    assert torch.allclose(sv_W, ev_S, atol=1e-3), (
        f"Eigenvalues of S don't match singular values of W.\n"
        f"max diff = {(sv_W - ev_S).abs().max():.3e}"
    )
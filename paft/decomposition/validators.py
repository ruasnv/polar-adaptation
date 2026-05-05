"""
Correctness validators for polar decomposition outputs.

These are called:
  - In test_polar.py to gate all experiments
  - Optionally at init time in PAFTModel to verify decomposition quality
  - In analysis scripts to audit saved tensors

All functions raise AssertionError on failure with a diagnostic message.
"""

import torch


def assert_orthogonal_columns(
    Q: torch.Tensor,
    atol: float = 1e-4,
    label: str = "Q",
) -> None:
    """
    Assert Q has orthonormal columns:  Q^T @ Q ≈ I_n

    Expected shape: [m, n] with m >= n  (e.g. Q_V [n_embd, d_head]).
    """
    assert Q.ndim == 2, f"{label} must be 2-D, got {Q.shape}"
    n = Q.shape[1]
    QtQ = Q.T @ Q
    I = torch.eye(n, device=Q.device, dtype=Q.dtype)
    max_dev = (QtQ - I).abs().max().item()
    assert max_dev < atol, (
        f"{label}: column orthogonality check failed — "
        f"max |Q^T Q - I| = {max_dev:.3e}  (atol={atol})"
    )


def assert_orthogonal_rows(
    Q: torch.Tensor,
    atol: float = 1e-4,
    label: str = "Q",
) -> None:
    """
    Assert Q has orthonormal rows:  Q @ Q^T ≈ I_m

    Expected shape: [m, n] with m <= n  (e.g. Q_O [d_head, n_embd]).
    """
    assert Q.ndim == 2, f"{label} must be 2-D, got {Q.shape}"
    m = Q.shape[0]
    QQt = Q @ Q.T
    I = torch.eye(m, device=Q.device, dtype=Q.dtype)
    max_dev = (QQt - I).abs().max().item()
    assert max_dev < atol, (
        f"{label}: row orthogonality check failed — "
        f"max |Q Q^T - I| = {max_dev:.3e}  (atol={atol})"
    )


def assert_symmetric(
    S: torch.Tensor,
    atol: float = 1e-4,
    label: str = "S",
) -> None:
    """Assert S is symmetric:  S ≈ S^T"""
    assert S.ndim == 2 and S.shape[0] == S.shape[1], (
        f"{label} must be square 2-D, got {S.shape}"
    )
    max_dev = (S - S.T).abs().max().item()
    assert max_dev < atol, (
        f"{label}: symmetry check failed — "
        f"max |S - S^T| = {max_dev:.3e}  (atol={atol})"
    )


def assert_psd(
    S: torch.Tensor,
    atol: float = 1e-4,
    label: str = "S",
) -> None:
    """Assert S is positive semidefinite: all eigenvalues >= -atol."""
    assert S.ndim == 2 and S.shape[0] == S.shape[1], (
        f"{label} must be square 2-D, got {S.shape}"
    )
    eigvals = torch.linalg.eigvalsh(S)
    min_ev = eigvals.min().item()
    assert min_ev >= -atol, (
        f"{label}: PSD check failed — "
        f"min eigenvalue = {min_ev:.3e}  (atol={atol})"
    )


def assert_symmetric_psd(
    S: torch.Tensor,
    atol: float = 1e-4,
    label: str = "S",
) -> None:
    """Assert S is symmetric and positive semidefinite."""
    assert_symmetric(S, atol=atol, label=label)
    assert_psd(S, atol=atol, label=label)


def assert_reconstruction(
    W: torch.Tensor,
    W_reconstructed: torch.Tensor,
    atol: float = 1e-4,
    label: str = "W",
) -> None:
    """Assert the reconstructed matrix matches the original within tolerance."""
    max_dev = (W - W_reconstructed).abs().max().item()
    assert max_dev < atol, (
        f"{label}: reconstruction check failed — "
        f"max |W - Q@S| = {max_dev:.3e}  (atol={atol})"
    )
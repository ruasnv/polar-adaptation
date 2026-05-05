"""
Tests for weight reconstruction in PAFTAttention and SVFAttention.

Verifies that reconstruct_W_V / reconstruct_W_O produce weights that
numerically match the original pretrained weights extracted from GPT-2.

Uses tiny synthetic models (not real GPT-2) for speed — the architecture
is identical, just with d_model=64, n_heads=4, d_head=16.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from paft.decomposition.polar import polar_decompose, polar_decompose_left
from paft.model.paft_model import PAFTAttention
from paft.model.svf_model import SVFAttention

ATOL = 1e-4

# Tiny GPT-2-like dims for fast tests
H  = 4      # n_heads
N  = 64     # n_embd
D  = 16     # d_head  (N // H)
MAX_POS = 128


def _make_config(n_heads=H, n_embd=N, max_position_embeddings=MAX_POS):
    return SimpleNamespace(
        n_head=n_heads,
        n_embd=n_embd,
        max_position_embeddings=max_position_embeddings,
    )


def _random_weights(seed=0):
    """Generate random W_V and W_O for all heads in one layer."""
    torch.manual_seed(seed)
    # W_V_h [N, D] per head → stacked [H, N, D]
    W_V = torch.stack([torch.randn(N, D) for _ in range(H)])
    # W_O_h [D, N] per head → stacked [H, D, N]
    W_O = torch.stack([torch.randn(D, N) for _ in range(H)])
    return W_V, W_O


def _decompose_all_heads(W_V, W_O):
    """Run polar decomposition on all heads, return stacked buffers."""
    Q_V_l, S_V_l, EV_V_l, lam_V_l = [], [], [], []
    Q_O_l, S_O_l, EV_O_l, lam_O_l = [], [], [], []

    for h in range(H):
        Q_V_h, S_V_h = polar_decompose(W_V[h].float())
        lv, EV_V_h   = torch.linalg.eigh(S_V_h)
        Q_V_l.append(Q_V_h);  S_V_l.append(S_V_h)
        EV_V_l.append(EV_V_h.flip(1));  lam_V_l.append(lv.flip(0))

        S_O_h, Q_O_h = polar_decompose_left(W_O[h].float())
        lo, EV_O_h   = torch.linalg.eigh(S_O_h)
        Q_O_l.append(Q_O_h);  S_O_l.append(S_O_h)
        EV_O_l.append(EV_O_h.flip(1));  lam_O_l.append(lo.flip(0))

    return (
        torch.stack(Q_V_l),  torch.stack(S_V_l),
        torch.stack(EV_V_l), torch.stack(lam_V_l),
        torch.stack(Q_O_l),  torch.stack(S_O_l),
        torch.stack(EV_O_l), torch.stack(lam_O_l),
    )


def _make_paft_attention(W_V, W_O, mode="hybrid"):
    Q_V, S_V, EV_V, lam_V, Q_O, S_O, EV_O, lam_O = _decompose_all_heads(W_V, W_O)
    cfg = _make_config()
    return PAFTAttention(
        config=cfg, layer_idx=0,
        Q_V=Q_V, Q_O=Q_O, EV_V=EV_V, EV_O=EV_O,
        S_V=S_V, S_O=S_O, lam_V=lam_V, lam_O=lam_O,
        W_Q=torch.randn(N, N), W_K=torch.randn(N, N),
        b_qkv=torch.zeros(3*N), b_o=torch.zeros(N),
        mode=mode,
    )


def _original_W_V_full(W_V):
    """Reconstruct full [N, N] W_V from per-head tensors — ground truth."""
    # Same reshape as PAFTAttention.reconstruct_W_V
    return W_V.permute(1, 0, 2).reshape(N, N)

def _original_W_O_full(W_O):
    """Reconstruct full [N, N] W_O from per-head tensors — ground truth."""
    return W_O.reshape(N, N)


# ──────────────────────────────────────────────────────────────────────────────
# PAFTAttention reconstruction
# ──────────────────────────────────────────────────────────────────────────────

class TestPAFTReconstruction:

    def test_hybrid_W_V_matches_original(self):
        W_V, W_O = _random_weights()
        attn = _make_paft_attention(W_V, W_O, mode="hybrid")
        W_V_rec = attn.reconstruct_W_V()
        W_V_orig = _original_W_V_full(W_V)
        max_err = (W_V_rec - W_V_orig).abs().max().item()
        assert max_err < ATOL, f"W_V reconstruction error {max_err:.3e} > {ATOL}"

    def test_hybrid_W_O_matches_original(self):
        W_V, W_O = _random_weights()
        attn = _make_paft_attention(W_V, W_O, mode="hybrid")
        W_O_rec = attn.reconstruct_W_O()
        W_O_orig = _original_W_O_full(W_O)
        max_err = (W_O_rec - W_O_orig).abs().max().item()
        assert max_err < ATOL, f"W_O reconstruction error {max_err:.3e} > {ATOL}"

    def test_pure_W_V_matches_original(self):
        """Pure mode: S built from EV @ diag(lam) @ EV.T should equal original S."""
        W_V, W_O = _random_weights()
        attn = _make_paft_attention(W_V, W_O, mode="pure")
        W_V_rec = attn.reconstruct_W_V()
        W_V_orig = _original_W_V_full(W_V)
        max_err = (W_V_rec - W_V_orig).abs().max().item()
        assert max_err < ATOL, f"Pure W_V reconstruction error {max_err:.3e}"

    def test_pure_W_O_matches_original(self):
        W_V, W_O = _random_weights()
        attn = _make_paft_attention(W_V, W_O, mode="pure")
        W_O_rec = attn.reconstruct_W_O()
        W_O_orig = _original_W_O_full(W_O)
        max_err = (W_O_rec - W_O_orig).abs().max().item()
        assert max_err < ATOL, f"Pure W_O reconstruction error {max_err:.3e}"

    def test_per_head_shapes(self):
        W_V, W_O = _random_weights()
        attn = _make_paft_attention(W_V, W_O)
        assert attn.get_W_V_per_head().shape == (H, N, D)
        assert attn.get_W_O_per_head().shape == (H, D, N)

    def test_full_shape(self):
        W_V, W_O = _random_weights()
        attn = _make_paft_attention(W_V, W_O)
        assert attn.reconstruct_W_V().shape == (N, N)
        assert attn.reconstruct_W_O().shape == (N, N)

    def test_per_head_consistent_with_full(self):
        """get_W_V_per_head and reconstruct_W_V must be consistent."""
        W_V, W_O = _random_weights()
        attn = _make_paft_attention(W_V, W_O)
        W_V_full = attn.reconstruct_W_V()
        W_V_heads = attn.get_W_V_per_head()   # [H, N, D]
        # Full is heads permuted and reshaped — reconstruct and compare
        W_V_from_heads = W_V_heads.permute(1, 0, 2).reshape(N, N)
        assert torch.allclose(W_V_full, W_V_from_heads, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# SVFAttention reconstruction
# ──────────────────────────────────────────────────────────────────────────────

class TestSVFReconstruction:

    def _make_svf_attention(self, W_V, W_O):
        U_V_l, sigma_V_l, Vh_V_l = [], [], []
        U_O_l, sigma_O_l, Vh_O_l = [], [], []
        for h in range(H):
            Uv, sv, Vhv = torch.linalg.svd(W_V[h].float(), full_matrices=False)
            U_V_l.append(Uv);  sigma_V_l.append(sv);  Vh_V_l.append(Vhv)
            Uo, so, Vho = torch.linalg.svd(W_O[h].float(), full_matrices=False)
            U_O_l.append(Uo);  sigma_O_l.append(so);  Vh_O_l.append(Vho)

        cfg = _make_config()
        return SVFAttention(
            config=cfg, layer_idx=0,
            U_V=torch.stack(U_V_l),  sigma_V=torch.stack(sigma_V_l),
            Vh_V=torch.stack(Vh_V_l),
            U_O=torch.stack(U_O_l),  sigma_O=torch.stack(sigma_O_l),
            Vh_O=torch.stack(Vh_O_l),
            W_Q=torch.randn(N, N),   W_K=torch.randn(N, N),
            b_qkv=torch.zeros(3*N),  b_o=torch.zeros(N),
        )

    def test_W_V_matches_original(self):
        W_V, W_O = _random_weights(seed=1)
        attn = self._make_svf_attention(W_V, W_O)
        W_V_rec = attn.reconstruct_W_V()
        W_V_orig = _original_W_V_full(W_V)
        max_err = (W_V_rec - W_V_orig).abs().max().item()
        assert max_err < ATOL, f"SVF W_V error {max_err:.3e}"

    def test_W_O_matches_original(self):
        W_V, W_O = _random_weights(seed=1)
        attn = self._make_svf_attention(W_V, W_O)
        W_O_rec = attn.reconstruct_W_O()
        W_O_orig = _original_W_O_full(W_O)
        max_err = (W_O_rec - W_O_orig).abs().max().item()
        assert max_err < ATOL, f"SVF W_O error {max_err:.3e}"

    def test_per_head_shapes(self):
        W_V, W_O = _random_weights(seed=1)
        attn = self._make_svf_attention(W_V, W_O)
        assert attn.get_W_V_per_head().shape == (H, N, D)
        assert attn.get_W_O_per_head().shape == (H, D, N)
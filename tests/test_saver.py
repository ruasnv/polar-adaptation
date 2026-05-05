"""
Tests for paft/checkpointing/saver.py and schema.py

Verifies that:
  1. All schema-required files are written to disk after each save event
  2. Validators correctly reject incomplete schemas
  3. F.ADAPTED_WEIGHTS (W_V, W_O) is saved in final/
  4. PAFT snapshot is saved for PAFT methods, absent for baselines
  5. Sentinel file marks completion
  6. Loader reads back what saver wrote
"""

import json
import pytest
import tempfile
from pathlib import Path

import torch

from paft.checkpointing.schema import (
    F, InitSchema, EpochSchema, FinalSchema,
    validate_init_schema, validate_epoch_schema, validate_final_schema,
)
from paft.checkpointing.saver import CheckpointSaver
from paft.checkpointing.loader import CheckpointLoader


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_run_dir(tmp_path):
    return tmp_path / "gpt2_small" / "news" / "hybrid_paft"


def _minimal_init_schema(method_name="hybrid_paft"):
    """Build the smallest valid InitSchema for a given method."""
    health = {"global": {"W_V": {"stable_rank": 1.0}, "W_O": {"stable_rank": 1.0}}}
    config = {"training": {"epochs": 3}}

    _PAFT = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
    _SVF  = {"svf"}

    if method_name in _PAFT:
        decomp = {
            "W_V_init": [torch.zeros(4, 8, 2)],
            "W_O_init": [torch.zeros(4, 2, 8)],
            "Q_V_0":    [torch.zeros(4, 8, 2)],
            "Q_O_0":    [torch.zeros(4, 2, 8)],
            "S_V_0":    [torch.zeros(4, 2, 2)],
            "S_O_0":    [torch.zeros(4, 2, 2)],
            "EV_V_0":   [torch.zeros(4, 2, 2)],
            "EV_O_0":   [torch.zeros(4, 2, 2)],
            "lam_V_0":  [torch.zeros(4, 2)],
            "lam_O_0":  [torch.zeros(4, 2)],
        }
    elif method_name in _SVF:
        decomp = {
            "W_V_init":  [torch.zeros(4, 8, 2)],
            "W_O_init":  [torch.zeros(4, 2, 8)],
            "U_V_0":     [torch.zeros(4, 8, 2)],
            "sigma_V_0": [torch.zeros(4, 2)],
            "Vh_V_0":    [torch.zeros(4, 2, 2)],
            "U_O_0":     [torch.zeros(4, 2, 2)],
            "sigma_O_0": [torch.zeros(4, 2)],
            "Vh_O_0":    [torch.zeros(4, 2, 8)],
        }
    else:
        decomp = None

    return InitSchema(config=config, geometric_health=health, decomp_init=decomp)


def _minimal_epoch_schema(epoch=0, method_name="hybrid_paft"):
    health = {"global": {}}
    paft_snap = None
    _PAFT = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
    if method_name in _PAFT:
        paft_snap = _dummy_paft_snapshot()
    return EpochSchema(
        epoch=epoch,
        metrics={"eval_loss": 2.5, "train_loss": 2.8},
        geometric_health=health,
        model_state={"dummy": torch.tensor(1.0)},
        optimizer_state={"state": {}},
        scheduler_state={"last_epoch": epoch},
        paft_snapshot=paft_snap,
    )


def _dummy_paft_snapshot():
    """Minimal PAFTSnapshot-like object (plain dict for test simplicity)."""
    from paft.methods.base import PAFTSnapshot
    snap = PAFTSnapshot()
    snap.Q_V   = [torch.zeros(4, 8, 2)]
    snap.Q_O   = [torch.zeros(4, 2, 8)]
    snap.S_V   = [torch.zeros(4, 2, 2)]
    snap.S_O   = [torch.zeros(4, 2, 2)]
    snap.EV_V  = [torch.zeros(4, 2, 2)]
    snap.EV_O  = [torch.zeros(4, 2, 2)]
    snap.lam_V = [torch.zeros(4, 2)]
    snap.lam_O = [torch.zeros(4, 2)]
    return snap


def _minimal_final_schema(method_name="hybrid_paft"):
    health = {"global": {}}
    _PAFT = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
    paft_snap = _dummy_paft_snapshot() if method_name in _PAFT else None
    return FinalSchema(
        metrics={"eval_loss": 2.3, "train_loss": 2.5},
        geometric_health=health,
        model_state={"dummy": torch.tensor(1.0)},
        adapted_weights={
            "W_V": [torch.zeros(4, 8, 2)],
            "W_O": [torch.zeros(4, 2, 8)],
        },
        paft_snapshot=paft_snap,
    )


# ──────────────────────────────────────────────────────────────────────────────
# InitSchema validator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestInitSchemaValidator:

    def test_accepts_valid_paft(self):
        schema = _minimal_init_schema("hybrid_paft")
        validate_init_schema(schema, "hybrid_paft")  # no raise

    def test_accepts_valid_svf(self):
        schema = _minimal_init_schema("svf")
        validate_init_schema(schema, "svf")

    def test_accepts_valid_baseline(self):
        schema = _minimal_init_schema("frozen")
        validate_init_schema(schema, "frozen")

    def test_rejects_paft_missing_decomp_init(self):
        schema = InitSchema(
            config={"a": 1},
            geometric_health={"g": 1},
            decomp_init=None,
        )
        with pytest.raises(AssertionError, match="decomp_init"):
            validate_init_schema(schema, "pure_paft")

    def test_rejects_paft_missing_key(self):
        schema = _minimal_init_schema("hybrid_paft")
        del schema.decomp_init["Q_V_0"]
        with pytest.raises(AssertionError, match="Q_V_0"):
            validate_init_schema(schema, "hybrid_paft")

    def test_rejects_empty_config(self):
        schema = InitSchema(config={}, geometric_health={"g": 1})
        with pytest.raises(AssertionError, match="config"):
            validate_init_schema(schema, "frozen")


# ──────────────────────────────────────────────────────────────────────────────
# EpochSchema validator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEpochSchemaValidator:

    def test_rejects_paft_missing_snapshot(self):
        schema = _minimal_epoch_schema(method_name="hybrid_paft")
        schema.paft_snapshot = None
        with pytest.raises(AssertionError, match="paft_snapshot"):
            validate_epoch_schema(schema, "hybrid_paft")

    def test_accepts_baseline_without_snapshot(self):
        schema = _minimal_epoch_schema(method_name="frozen")
        validate_epoch_schema(schema, "frozen")


# ──────────────────────────────────────────────────────────────────────────────
# FinalSchema validator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFinalSchemaValidator:

    def test_rejects_missing_adapted_weights(self):
        schema = _minimal_final_schema()
        schema.adapted_weights = {}
        with pytest.raises(AssertionError, match="adapted_weights"):
            validate_final_schema(schema, "hybrid_paft")

    def test_rejects_missing_W_V_key(self):
        schema = _minimal_final_schema()
        del schema.adapted_weights["W_V"]
        with pytest.raises(AssertionError, match="W_V"):
            validate_final_schema(schema, "frozen")

    def test_rejects_paft_without_snapshot(self):
        schema = _minimal_final_schema("pure_paft")
        schema.paft_snapshot = None
        with pytest.raises(AssertionError, match="paft_snapshot"):
            validate_final_schema(schema, "pure_paft")

    def test_accepts_valid_final(self):
        schema = _minimal_final_schema("hybrid_paft")
        validate_final_schema(schema, "hybrid_paft")


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointSaver — file existence tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSaverFilesOnDisk:

    def test_save_init_creates_required_files(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "hybrid_paft")
        saver.save_init(_minimal_init_schema("hybrid_paft"))

        init_dir = tmp_run_dir / "init"
        assert (init_dir / F.CONFIG).exists(),          "config.json missing"
        assert (init_dir / F.GEOMETRIC_HEALTH).exists(),"geometric_health.pt missing"
        assert (init_dir / F.DECOMP_INIT).exists(),     "decomp_init.pt missing (PAFT)"

    def test_save_init_no_decomp_for_baseline(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        saver.save_init(_minimal_init_schema("frozen"))
        assert not (tmp_run_dir / "init" / F.DECOMP_INIT).exists()

    def test_save_epoch_creates_required_files(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "hybrid_paft")
        saver.save_epoch(_minimal_epoch_schema(epoch=0, method_name="hybrid_paft"))

        epoch_dir = tmp_run_dir / "epoch_0"
        assert (epoch_dir / F.METRICS).exists()
        assert (epoch_dir / F.GEOMETRIC_HEALTH).exists()
        assert (epoch_dir / F.MODEL).exists()
        assert (epoch_dir / F.OPTIMIZER).exists()
        assert (epoch_dir / F.SCHEDULER).exists()
        assert (epoch_dir / F.PAFT_SNAPSHOT).exists()

    def test_save_epoch_no_snapshot_for_baseline(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        saver.save_epoch(_minimal_epoch_schema(epoch=0, method_name="frozen"))
        assert not (tmp_run_dir / "epoch_0" / F.PAFT_SNAPSHOT).exists()

    def test_save_final_creates_required_files(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "hybrid_paft")
        saver.save_final(_minimal_final_schema("hybrid_paft"))

        final_dir = tmp_run_dir / "final"
        assert (final_dir / F.METRICS).exists()
        assert (final_dir / F.GEOMETRIC_HEALTH).exists()
        assert (final_dir / F.MODEL).exists()
        assert (final_dir / F.ADAPTED_WEIGHTS).exists(),  "adapted_weights.pt missing"
        assert (final_dir / F.PAFT_SNAPSHOT).exists()
        assert (final_dir / F.SENTINEL).exists(),         "training_complete sentinel missing"

    def test_save_final_adapted_weights_for_baseline(self, tmp_run_dir):
        """Adapted weights must be saved for ALL methods, not just PAFT."""
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        saver.save_final(_minimal_final_schema("frozen"))
        assert (tmp_run_dir / "final" / F.ADAPTED_WEIGHTS).exists()

    def test_is_complete_after_save_final(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        assert not saver.is_complete()
        saver.save_final(_minimal_final_schema("frozen"))
        assert saver.is_complete()


# ──────────────────────────────────────────────────────────────────────────────
# Loader round-trip tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLoaderRoundTrip:

    def test_config_round_trip(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        cfg = {"training": {"epochs": 5, "learning_rate": 2e-4}}
        saver.save_init(InitSchema(
            config=cfg,
            geometric_health={"g": 1.0},
        ))
        loader = CheckpointLoader(tmp_run_dir)
        loaded = loader.load_init_config()
        assert loaded["training"]["epochs"] == 5

    def test_metrics_round_trip(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        saver.save_epoch(_minimal_epoch_schema(epoch=0, method_name="frozen"))
        loader = CheckpointLoader(tmp_run_dir)
        metrics = loader.load_epoch_metrics(0)
        assert metrics["eval_loss"] == pytest.approx(2.5)

    def test_adapted_weights_round_trip(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        W_V = [torch.randn(4, 8, 2)]
        W_O = [torch.randn(4, 2, 8)]
        schema = _minimal_final_schema("frozen")
        schema.adapted_weights = {"W_V": W_V, "W_O": W_O}
        saver.save_final(schema)

        loader = CheckpointLoader(tmp_run_dir)
        aw = loader.load_adapted_weights()
        assert torch.allclose(aw["W_V"][0], W_V[0], atol=1e-6)
        assert torch.allclose(aw["W_O"][0], W_O[0], atol=1e-6)

    def test_n_epochs_saved(self, tmp_run_dir):
        saver = CheckpointSaver(tmp_run_dir, "frozen")
        for e in range(3):
            saver.save_epoch(_minimal_epoch_schema(epoch=e, method_name="frozen"))
        loader = CheckpointLoader(tmp_run_dir)
        assert loader.n_epochs_saved() == 3
"""
Checkpoint schema — defines exactly what is saved at each training event.

ONE-SHOT TRAINING CONSTRAINT
─────────────────────────────
Each of the 100 experiment runs is trained exactly once.  If a tensor
is not in this schema, it cannot be recovered without re-running.  Any
analysis script that needs a tensor must have it listed here.

Before modifying this schema, check analysis/ for every script that
reads checkpoints and ensure its required tensors are present.

SAVE EVENTS
───────────
Three events trigger saves:

  INIT     Before the first training step.
           Captures pretrained state — reference for all geometric deltas.

  EPOCH    After each epoch's eval pass completes.
           One directory per epoch under {run_dir}/epoch_{n}/.
           Contains: task metrics, geometric health, model + optimizer state,
           PAFT snapshot (if applicable).

  FINAL    After the last epoch.
           Duplicate of the last EPOCH save plus an explicit 'final' marker.
           Separating FINAL from EPOCH makes analysis scripts simpler —
           they can always load from 'final/' without knowing n_epochs.

DIRECTORY STRUCTURE
────────────────────
results/checkpoints/{model}/{domain}/{method}/
    init/
        config.json            full merged config
        geometric_health.pt    pretrained baseline metrics
        decomp_init.pt         decomposition tensors at t=0  (PAFT/SVF only)
    epoch_0/
        metrics.json           task metrics after epoch 0
        geometric_health.pt    per-layer health at end of epoch
        model.pt               full model state_dict
        optimizer.pt           optimizer state_dict
        scheduler.pt           scheduler state_dict
        paft_snapshot.pt       Q, S, EV, lam per layer  (PAFT only)
    epoch_1/
        ...
    final/
        metrics.json
        geometric_health.pt
        model.pt
        adapted_weights.pt     W_V and W_O per head (all methods)
        paft_snapshot.pt       (PAFT only)
        training_complete      empty sentinel file

WHAT EACH ANALYSIS SCRIPT NEEDS
─────────────────────────────────
geometric_health.py    → epoch_*/geometric_health.pt  (all methods)
geometric_audit.py     → init/decomp_init.pt + epoch_*/paft_snapshot.pt  (PAFT)
domain_correlation.py  → init/decomp_init.pt + final/paft_snapshot.pt  (PAFT)
layer_profiles.py      → epoch_*/paft_snapshot.pt  (PAFT)
eigenvalue_semantics.py→ final/adapted_weights.pt + final/paft_snapshot.pt  (PAFT)
residual_stability.py  → final/adapted_weights.pt  (all methods)
dial_ablation.py       → epoch_*/metrics.json  (hybrid_paft variants)
efficiency_curve.py    → final/metrics.json + config.json  (all methods)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Filenames — single source of truth used by both saver and loader
# ──────────────────────────────────────────────────────────────────────────────

class F:
    """Canonical filenames.  Never hardcode these elsewhere."""
    CONFIG           = "config.json"
    METRICS          = "metrics.json"
    GEOMETRIC_HEALTH = "geometric_health.pt"
    DECOMP_INIT      = "decomp_init.pt"
    MODEL            = "model.pt"
    OPTIMIZER        = "optimizer.pt"
    SCHEDULER        = "scheduler.pt"
    PAFT_SNAPSHOT    = "paft_snapshot.pt"
    ADAPTED_WEIGHTS  = "adapted_weights.pt"   # W_V and W_O per head at final
    SENTINEL         = "training_complete"


# ──────────────────────────────────────────────────────────────────────────────
# Schema dataclasses — typed containers for each save event
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InitSchema:
    """
    Saved once before training begins.
    All tensors are the pretrained model's values — reference for deltas.
    """
    config:           Dict[str, Any]       # full merged config
    geometric_health: Dict[str, Any]       # pretrained geometric health snapshot
    decomp_init:      Optional[Dict[str, Any]] = None
    # decomp_init keys (PAFT): W_V_init, W_O_init, Q_V_0, Q_O_0, S_V_0, S_O_0,
    #                          EV_V_0, EV_O_0, lam_V_0, lam_O_0
    # decomp_init keys (SVF):  W_V_init, W_O_init, U_V_0, sigma_V_0, Vh_V_0,
    #                          U_O_0, sigma_O_0, Vh_O_0
    # decomp_init = None for frozen, full_finetune, bitfit, lora, polar


@dataclass
class EpochSchema:
    """
    Saved after each epoch's eval pass.
    model_state, optimizer_state, scheduler_state are for training resumption.
    geometric_health and paft_snapshot are for post-training analysis.
    """
    epoch:            int
    metrics:          Dict[str, float]     # task metrics: loss, accuracy/rouge/f1
    geometric_health: Dict[str, Any]       # per-layer health snapshot
    model_state:      Dict[str, Any]       # model.state_dict()
    optimizer_state:  Dict[str, Any]       # optimizer.state_dict()
    scheduler_state:  Dict[str, Any]       # scheduler.state_dict()
    paft_snapshot:    Optional[Any] = None # PAFTSnapshot dataclass | None


@dataclass
class FinalSchema:
    """
    Saved after the last epoch.  Mirror of the last EpochSchema plus sentinel.
    Analysis scripts load from 'final/' — no need to know n_epochs.

    adapted_weights contains the live reconstructed W_V and W_O per head,
    for every layer.  Required by eigenvalue_semantics.py and
    residual_stability.py — both need the actual weights, not decomposition
    components.  Stored as CPU tensors to avoid re-loading the full model.

    adapted_weights keys:
        "W_V": List[n_layers] of Tensor[n_heads, n_embd, d_head]
        "W_O": List[n_layers] of Tensor[n_heads, d_head, n_embd]
    """
    metrics:          Dict[str, float]
    geometric_health: Dict[str, Any]
    model_state:      Dict[str, Any]
    adapted_weights:  Dict[str, List]          # W_V and W_O per head, CPU
    paft_snapshot:    Optional[Any] = None


def validate_final_schema(schema: FinalSchema, method_name: str) -> None:
    """Assert FinalSchema completeness before writing to disk."""
    assert schema.metrics,          "FinalSchema.metrics is empty"
    assert schema.geometric_health, "FinalSchema.geometric_health is empty"
    assert schema.model_state,      "FinalSchema.model_state is empty"
    assert schema.adapted_weights,  "FinalSchema.adapted_weights is empty"
    assert "W_V" in schema.adapted_weights, "adapted_weights missing 'W_V'"
    assert "W_O" in schema.adapted_weights, "adapted_weights missing 'W_O'"
    assert len(schema.adapted_weights["W_V"]) > 0, "W_V list is empty"
    assert len(schema.adapted_weights["W_O"]) > 0, "W_O list is empty"

    _PAFT_METHODS = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
    if method_name in _PAFT_METHODS:
        assert schema.paft_snapshot is not None, (
            f"Method '{method_name}' requires FinalSchema.paft_snapshot"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Schema validator — call before training starts to catch missing tensors early
# ──────────────────────────────────────────────────────────────────────────────

def validate_init_schema(schema: InitSchema, method_name: str) -> None:
    """
    Assert that InitSchema contains everything the method requires.
    Call this at the start of training, before any epochs run.
    Better to crash here than to finish 5 epochs and discover a missing tensor.
    """
    assert schema.config,           "InitSchema.config is empty"
    assert schema.geometric_health, "InitSchema.geometric_health is empty"

    _PAFT_METHODS = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
    _SVF_METHODS  = {"svf"}

    if method_name in _PAFT_METHODS:
        assert schema.decomp_init is not None, (
            f"Method '{method_name}' requires InitSchema.decomp_init — got None"
        )
        required_keys = {
            "W_V_init", "W_O_init",
            "Q_V_0", "Q_O_0",
            "S_V_0", "S_O_0",
            "EV_V_0", "EV_O_0",
            "lam_V_0", "lam_O_0",
        }
        missing = required_keys - set(schema.decomp_init.keys())
        assert not missing, f"decomp_init missing keys for {method_name}: {missing}"

    elif method_name in _SVF_METHODS:
        assert schema.decomp_init is not None, (
            f"Method '{method_name}' requires InitSchema.decomp_init — got None"
        )
        required_keys = {
            "W_V_init", "W_O_init",
            "U_V_0", "sigma_V_0", "Vh_V_0",
            "U_O_0", "sigma_O_0", "Vh_O_0",
        }
        missing = required_keys - set(schema.decomp_init.keys())
        assert not missing, f"decomp_init missing keys for {method_name}: {missing}"


def validate_epoch_schema(schema: EpochSchema, method_name: str) -> None:
    """Assert EpochSchema completeness before writing to disk."""
    assert isinstance(schema.epoch, int) and schema.epoch >= 0
    assert schema.metrics,          "EpochSchema.metrics is empty"
    assert schema.geometric_health, "EpochSchema.geometric_health is empty"
    assert schema.model_state,      "EpochSchema.model_state is empty"
    assert schema.optimizer_state,  "EpochSchema.optimizer_state is empty"
    assert schema.scheduler_state,  "EpochSchema.scheduler_state is empty"

    _PAFT_METHODS = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
    if method_name in _PAFT_METHODS:
        assert schema.paft_snapshot is not None, (
            f"Method '{method_name}' requires EpochSchema.paft_snapshot"
        )
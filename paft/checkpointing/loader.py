"""
CheckpointLoader — reads checkpoints for analysis scripts and training resumption.

Analysis scripts in analysis/ load from here — never call torch.load directly.
This keeps the loading logic centralised so filename changes only happen once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from paft.checkpointing.schema import F

logger = logging.getLogger(__name__)


class CheckpointLoader:
    """
    Read-only access to one experiment run's checkpoints.

    Usage (analysis script):
        loader = CheckpointLoader(run_dir)
        init_health = loader.load_init_geometric_health()
        for epoch in range(loader.n_epochs_saved()):
            health = loader.load_epoch_geometric_health(epoch)
            snap   = loader.load_epoch_paft_snapshot(epoch)
    """

    def __init__(self, run_dir: Path | str, device: str = "cpu") -> None:
        self.run_dir = Path(run_dir)
        self.device  = device
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    # ── init ─────────────────────────────────────────────────────────────────

    def load_init_config(self) -> Dict[str, Any]:
        return _read_json(self.run_dir / "init" / F.CONFIG)

    def load_init_geometric_health(self) -> Dict[str, Any]:
        return _read_pt(self.run_dir / "init" / F.GEOMETRIC_HEALTH, self.device)

    def load_decomp_init(self) -> Optional[Dict[str, Any]]:
        """Returns None if not a surgery method (no decomp_init saved)."""
        path = self.run_dir / "init" / F.DECOMP_INIT
        return _read_pt(path, self.device) if path.exists() else None

    # ── epoch ────────────────────────────────────────────────────────────────

    def n_epochs_saved(self) -> int:
        """Count epoch_* directories that exist under run_dir."""
        return len(sorted(self.run_dir.glob("epoch_*")))

    def load_epoch_metrics(self, epoch: int) -> Dict[str, float]:
        return _read_json(self.run_dir / f"epoch_{epoch}" / F.METRICS)

    def load_epoch_geometric_health(self, epoch: int) -> Dict[str, Any]:
        return _read_pt(
            self.run_dir / f"epoch_{epoch}" / F.GEOMETRIC_HEALTH, self.device
        )

    def load_epoch_paft_snapshot(self, epoch: int) -> Optional[Dict[str, Any]]:
        path = self.run_dir / f"epoch_{epoch}" / F.PAFT_SNAPSHOT
        return _read_pt(path, self.device) if path.exists() else None

    def load_epoch_model_state(self, epoch: int) -> Dict[str, Any]:
        return _read_pt(
            self.run_dir / f"epoch_{epoch}" / F.MODEL, self.device
        )

    def load_epoch_training_state(self, epoch: int) -> Dict[str, Any]:
        """Load optimizer + scheduler state for training resumption."""
        d = self.run_dir / f"epoch_{epoch}"
        return {
            "optimizer": _read_pt(d / F.OPTIMIZER, self.device),
            "scheduler": _read_pt(d / F.SCHEDULER, self.device),
            "epoch":     epoch,
        }

    # ── final ────────────────────────────────────────────────────────────────

    def load_final_metrics(self) -> Dict[str, float]:
        return _read_json(self.run_dir / "final" / F.METRICS)

    def load_final_geometric_health(self) -> Dict[str, Any]:
        return _read_pt(self.run_dir / "final" / F.GEOMETRIC_HEALTH, self.device)

    def load_final_paft_snapshot(self) -> Optional[Dict[str, Any]]:
        path = self.run_dir / "final" / F.PAFT_SNAPSHOT
        return _read_pt(path, self.device) if path.exists() else None

    def load_final_model_state(self) -> Dict[str, Any]:
        return _read_pt(self.run_dir / "final" / F.MODEL, self.device)

    def load_adapted_weights(self) -> Dict[str, Any]:
        """
        Load W_V and W_O per head from final/.
        Returns {"W_V": List[n_layers × Tensor[H,n,d]], "W_O": List[...]}
        Used by eigenvalue_semantics.py and residual_stability.py.
        """
        return _read_pt(self.run_dir / "final" / F.ADAPTED_WEIGHTS, self.device)

    def is_complete(self) -> bool:
        return (self.run_dir / "final" / F.SENTINEL).exists()

    # ── convenience: all epochs for one field ────────────────────────────────

    def all_epoch_metrics(self) -> List[Dict[str, float]]:
        return [self.load_epoch_metrics(e) for e in range(self.n_epochs_saved())]

    def all_epoch_geometric_health(self) -> List[Dict[str, Any]]:
        return [
            self.load_epoch_geometric_health(e)
            for e in range(self.n_epochs_saved())
        ]

    def all_epoch_paft_snapshots(self) -> List[Optional[Dict[str, Any]]]:
        return [
            self.load_epoch_paft_snapshot(e)
            for e in range(self.n_epochs_saved())
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path | str, device: str = "cpu") -> CheckpointLoader:
    """One-liner for analysis scripts."""
    return CheckpointLoader(run_dir, device)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_pt(path: Path, device: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
"""
CheckpointSaver — writes InitSchema, EpochSchema, FinalSchema to disk.

Single source of truth for all checkpoint I/O.  The trainer calls this;
nothing else writes checkpoints directly.

All tensors are moved to CPU before saving — no VRAM held by checkpoints.
JSON is used for metrics and config (human-readable, grep-able).
torch.save is used for tensors and state dicts.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from paft.checkpointing.schema import (
    F, InitSchema, EpochSchema, FinalSchema,
    validate_init_schema, validate_epoch_schema, validate_final_schema,
)

logger = logging.getLogger(__name__)


class CheckpointSaver:
    """
    Manages all checkpoint writes for one experiment run.

    Usage:
        saver = CheckpointSaver(run_dir, method_name)
        saver.save_init(init_schema)
        for epoch in range(n_epochs):
            ...train...
            saver.save_epoch(epoch_schema)
        saver.save_final(final_schema)

    run_dir layout:
        {run_dir}/init/
        {run_dir}/epoch_0/
        {run_dir}/epoch_1/
        ...
        {run_dir}/final/
    """

    def __init__(self, run_dir: Path | str, method_name: str) -> None:
        self.run_dir     = Path(run_dir)
        self.method_name = method_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CheckpointSaver: run_dir={self.run_dir}")

    # ── save events ──────────────────────────────────────────────────────────

    def save_init(self, schema: InitSchema) -> None:
        """Save pretrained baseline state.  Call once before training starts."""
        validate_init_schema(schema, self.method_name)
        d = self.run_dir / "init"
        d.mkdir(exist_ok=True)

        _write_json(d / F.CONFIG, schema.config)
        _write_pt(d / F.GEOMETRIC_HEALTH, schema.geometric_health)

        if schema.decomp_init is not None:
            # Move all tensors to CPU before saving
            cpu_decomp = _tensors_to_cpu(schema.decomp_init)
            _write_pt(d / F.DECOMP_INIT, cpu_decomp)

        logger.info(f"Saved init checkpoint -> {d}")

    def save_epoch(self, schema: EpochSchema) -> None:
        """Save after each epoch.  Call after eval metrics are computed."""
        validate_epoch_schema(schema, self.method_name)
        d = self.run_dir / f"epoch_{schema.epoch}"
        d.mkdir(exist_ok=True)

        _write_json(d / F.METRICS, schema.metrics)
        _write_pt(d / F.GEOMETRIC_HEALTH, schema.geometric_health)
        _write_pt(d / F.MODEL,     schema.model_state)
        _write_pt(d / F.OPTIMIZER, schema.optimizer_state)
        _write_pt(d / F.SCHEDULER, schema.scheduler_state)

        if schema.paft_snapshot is not None:
            _write_pt(d / F.PAFT_SNAPSHOT, _snapshot_to_cpu(schema.paft_snapshot))

        logger.info(
            f"Saved epoch {schema.epoch} checkpoint -> {d}  "
            f"metrics={_fmt_metrics(schema.metrics)}"
        )

    def save_final(self, schema: FinalSchema) -> None:
        """Save final state.  Call after the last epoch's save_epoch."""
        validate_final_schema(schema, self.method_name)
        d = self.run_dir / "final"
        d.mkdir(exist_ok=True)

        _write_json(d / F.METRICS, schema.metrics)
        _write_pt(d / F.GEOMETRIC_HEALTH, schema.geometric_health)
        _write_pt(d / F.MODEL, schema.model_state)

        # Adapted weights — CPU tensors, required by analysis scripts
        cpu_adapted = {
            "W_V": [t.detach().cpu() for t in schema.adapted_weights["W_V"]],
            "W_O": [t.detach().cpu() for t in schema.adapted_weights["W_O"]],
        }
        _write_pt(d / F.ADAPTED_WEIGHTS, cpu_adapted)

        if schema.paft_snapshot is not None:
            _write_pt(d / F.PAFT_SNAPSHOT, _snapshot_to_cpu(schema.paft_snapshot))

        # Sentinel file — marks training as complete
        (d / F.SENTINEL).touch()

        logger.info(f"Saved final checkpoint -> {d}")

    # ── introspection ────────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        """Return True if the sentinel file exists (training finished)."""
        return (self.run_dir / "final" / F.SENTINEL).exists()

    def epoch_dir(self, epoch: int) -> Path:
        return self.run_dir / f"epoch_{epoch}"

    def init_dir(self) -> Path:
        return self.run_dir / "init"

    def final_dir(self) -> Path:
        return self.run_dir / "final"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _write_pt(path: Path, data: Any) -> None:
    torch.save(data, path)


def _json_default(obj: Any) -> Any:
    """Handle non-JSON-serialisable types (tensors, numpy scalars, etc.)."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _tensors_to_cpu(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively move tensors in a dict to CPU."""
    return {
        k: v.detach().cpu() if isinstance(v, torch.Tensor)
           else _tensors_to_cpu(v) if isinstance(v, dict)
           else v
        for k, v in d.items()
    }


def _snapshot_to_cpu(snap) -> Any:
    """
    Move a PAFTSnapshot's tensor lists to CPU.
    snap is a PAFTSnapshot dataclass with List[Tensor] fields.
    Returns a plain dict for torch.save.
    """
    if snap is None:
        return None
    return {
        field: [t.detach().cpu() for t in getattr(snap, field)]
        for field in ("Q_V", "Q_O", "S_V", "S_O", "EV_V", "EV_O", "lam_V", "lam_O")
        if hasattr(snap, field)
    }


def _fmt_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict for log line (first 3 keys only)."""
    items = list(metrics.items())[:3]
    return "  ".join(f"{k}={v:.4f}" for k, v in items)
"""
Device management and VRAM monitoring.

Kept deliberately minimal — just hardware detection and the memory heartbeat
used by the trainer to monitor VRAM pressure across the 100-run sweep.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# 90% of 8 GB expressed in MB.  Adjust if running on a different GPU.
_VRAM_WARN_MB: float = 7_300.0


def get_device() -> torch.device:
    """Return the best available device: CUDA if present, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name  = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info(f"Using GPU: {name} ({total:.0f} MB total VRAM)")
        return device
    logger.info("CUDA unavailable — running on CPU")
    return torch.device("cpu")


def log_vram_usage(prefix: str = "") -> None:
    """
    Log current VRAM allocation.  Called by the trainer at key checkpoints:
    before/after decomposition, start/end of each epoch.

    Does NOT call empty_cache() — that would distort the reserved-memory
    reading and belongs only at experiment boundaries (BaseMethod.cleanup).

    Logs a warning if peak allocation exceeds _VRAM_WARN_MB.
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated()  / (1024 ** 2)
    reserved  = torch.cuda.memory_reserved()   / (1024 ** 2)
    peak      = torch.cuda.max_memory_allocated() / (1024 ** 2)

    logger.info(
        f"[{prefix}] VRAM — allocated: {allocated:.1f} MB  "
        f"reserved: {reserved:.1f} MB  peak: {peak:.1f} MB"
    )

    if peak > _VRAM_WARN_MB:
        logger.warning(
            f"[{prefix}] VRAM CRITICAL: peak {peak:.1f} MB exceeds "
            f"{_VRAM_WARN_MB:.0f} MB threshold — OOM risk"
        )


def reset_peak_vram() -> None:
    """Reset peak VRAM counter.  Call at the start of each experiment."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
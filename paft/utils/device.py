from __future__ import annotations

import logging
import torch
import log_utils
import os

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Detect the best available hardware."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_vram_usage(prefix: str = ""):
    """
    The Memory Heartbeat.
    Essential for tracking the VRAM limits.
    """
    if not torch.cuda.is_available():
        return

    # Flush memory before checking
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_used = torch.cuda.max_memory_allocated() / (1024 ** 2)

    logger.info(
        f"[{prefix}] VRAM: {allocated:.1f}MB allocated, "
        f"{reserved:.1f}MB reserved, "
        f"Peak: {max_used:.1f}MB / 8192MB"
    )

    # Safety Check: If we exceed 90% of your 8GB, log a warning
    if max_used > 7300:
        logger.warning("VRAM CRITICAL: Vram approaching its limit!")


def set_seed(seed: int):
    """Ensure reproducibility across your 204 runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
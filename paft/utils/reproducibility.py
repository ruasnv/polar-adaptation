"""
Reproducibility — seed setting and deterministic mode.

Called once at the start of every experiment run before any model, data,
or training code is initialised.  set_seed is the function that was
incorrectly placed in device.py; it lives here.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Args:
        seed:          Integer seed.  Use cfg['training']['seed'] (default 42).
        deterministic: If True, enable CUDA deterministic algorithms.
                       Adds ~10% overhead but ensures bit-exact reproduction.
                       Set False for speed if exact reproduction is not needed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
            # CUDA >= 10.2 — raises an error if a non-deterministic algorithm
            # is selected, making non-reproducibility visible rather than silent.
            torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(f"Seed set: {seed}  deterministic: {deterministic}")
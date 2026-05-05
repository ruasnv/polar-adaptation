"""
Learning rate scheduling.

Linear warmup followed by linear decay to 0 — as specified in base.yaml.
No exotic schedules needed: the goal is a fair comparison across methods,
and a standard schedule eliminates scheduling as a confound.

  LR
  ^
  |    /-------+
  |   /         |
  |  /           |
  | /             |
  |/               +---
  +-------------------> step
      warmup   decay
"""

from __future__ import annotations

import logging
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def get_linear_schedule(
    optimizer:      Optimizer,
    n_warmup_steps: int,
    n_total_steps:  int,
) -> LambdaLR:
    """
    Linear warmup then linear decay to 0.

    Steps 0 … n_warmup_steps:    LR scales linearly from 0 to peak.
    Steps n_warmup_steps … n_total: LR scales linearly from peak to 0.

    Args:
        optimizer:      The optimizer whose LR groups will be scheduled.
        n_warmup_steps: Number of warmup steps (= total_steps * warmup_ratio).
        n_total_steps:  Total training steps (n_epochs * steps_per_epoch).

    Returns:
        LambdaLR scheduler.  Call scheduler.step() after every optimizer step.
    """
    if n_warmup_steps >= n_total_steps:
        raise ValueError(
            f"n_warmup_steps={n_warmup_steps} >= n_total_steps={n_total_steps}. "
            "Warmup must be shorter than total training."
        )

    def lr_lambda(current_step: int) -> float:
        if current_step < n_warmup_steps:
            return float(current_step) / float(max(1, n_warmup_steps))
        progress = float(current_step - n_warmup_steps) / float(
            max(1, n_total_steps - n_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    logger.info(
        f"Scheduler: linear warmup {n_warmup_steps} steps, "
        f"linear decay to {n_total_steps} steps"
    )
    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer:       Optimizer,
    cfg:             dict,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Build scheduler from config.  Called by the trainer.

    Reads:
        cfg['training']['epochs']        — total epochs
        cfg['training']['warmup_ratio']  — fraction of total steps for warmup
    """
    n_epochs      = cfg["training"]["epochs"]
    warmup_ratio  = cfg["training"].get("warmup_ratio", 0.06)
    n_total       = n_epochs * steps_per_epoch
    n_warmup      = max(1, int(n_total * warmup_ratio))
    return get_linear_schedule(optimizer, n_warmup, n_total)
"""
Trainer — the main training loop.

Designed to be completely method-agnostic.  It calls the BaseMethod interface
and nothing else.  No if-branches on method type anywhere in this file.

LOOP STRUCTURE
──────────────
for epoch in range(n_epochs):
    train_epoch()
        for batch in train_loader:
            loss = method.forward(input_ids, attention_mask, labels)
            loss.backward()
            method.pre_optimizer_step()          # no-op for all 11 methods
            clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    eval_epoch()
        for batch in val_loader:
            loss = method.forward(...)           # no_grad
    geometric_health_snapshot()                 # CPU SVD, once per epoch
    saver.save_epoch(...)
    log_vram_usage(...)

VRAM BUDGET (GPT-2 medium, hybrid_paft, batch=32, seq=512)
───────────────────────────────────────────────────────────
  Model weights (fp32): ~1.4 GB
  Trainable params + grad: ~0.01 GB  (S_V, S_O only)
  AdamW moment tensors: ~0.02 GB
  Activations (w/ grad checkpointing): ~1.5 GB
  Forward buffer: ~0.5 GB
  Total: ~3.4 GB  →  fits in 6 GB with headroom

DATA INTERFACE CONTRACT
───────────────────────
The trainer expects DataLoaders that yield dicts with at minimum:
    {"input_ids": Tensor[B,T], "attention_mask": Tensor[B,T], "labels": Tensor[B,T]}
This matches the output of the data modules in paft/data/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from paft.methods.base import BaseMethod
from paft.checkpointing.saver import CheckpointSaver
from paft.checkpointing.schema import (
    InitSchema, EpochSchema, FinalSchema,
    validate_init_schema,
)
from paft.training.scheduler import build_scheduler
from paft.utils.device import log_vram_usage, reset_peak_vram

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trains one method on one (model, domain, task) configuration.

    Instantiate once per experiment run.  Call train() to run all epochs.
    After training, call method.cleanup() to release VRAM.

    Args:
        method:         Built BaseMethod instance (build() already called).
        train_loader:   DataLoader yielding {"input_ids", "attention_mask", "labels"}.
        val_loader:     DataLoader for eval (no gradients computed).
        cfg:            Fully merged config dict from ConfigLoader.
        run_dir:        Directory for this run's checkpoints.
    """

    def __init__(
        self,
        method:       BaseMethod,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          Dict[str, Any],
        run_dir:      Path | str,
    ) -> None:
        self.method       = method
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.run_dir      = Path(run_dir)

        tcfg = cfg["training"]
        self.n_epochs     = tcfg["epochs"]
        self.grad_clip    = tcfg.get("gradient_clip", 1.0)
        self.eval_every   = tcfg.get("eval_every_n_steps", 500)
        self.log_every    = tcfg.get("log_every_n_steps", 10)

        self.device       = method.device
        self.method_name  = method.method_name

        # Optimizer and scheduler — built here so they're available to the saver
        self.optimizer = method.get_optimizer(
            lr           = tcfg["learning_rate"],
            weight_decay = tcfg.get("weight_decay", 0.01),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            cfg,
            steps_per_epoch = len(train_loader),
        )

        self.saver = CheckpointSaver(run_dir, self.method_name)
        self._global_step = 0

    # ── public entry point ───────────────────────────────────────────────────

    def train(self) -> Dict[str, float]:
        """
        Run all epochs.  Returns the final eval metrics dict.

        Saves:
            init/      before epoch 0
            epoch_N/   after each epoch's eval
            final/     after last epoch (mirrors last epoch_N)
        """
        logger.info(
            f"Training {self.method_name} for {self.n_epochs} epochs  "
            f"({self.method.num_trainable_params():,} trainable params)"
        )

        reset_peak_vram()
        log_vram_usage(f"{self.method_name}/start")

        # ── init checkpoint ──────────────────────────────────────────────────
        init_schema = self._build_init_schema()
        validate_init_schema(init_schema, self.method_name)
        self.saver.save_init(init_schema)

        final_metrics: Dict[str, float] = {}

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            logger.info(f"── Epoch {epoch}/{self.n_epochs - 1} ──")

            train_loss = self._train_epoch(epoch)
            eval_metrics = self._eval_epoch(epoch)
            eval_metrics["train_loss"] = train_loss
            eval_metrics["epoch"]      = float(epoch)

            # Geometric health — CPU SVD, no VRAM impact
            health = self.method.geometric_health_snapshot()

            epoch_schema = EpochSchema(
                epoch            = epoch,
                metrics          = eval_metrics,
                geometric_health = _serialise_health(health),
                model_state      = self.method.state_dict()["model"],
                optimizer_state  = self.optimizer.state_dict(),
                scheduler_state  = self.scheduler.state_dict(),
                paft_snapshot    = self.method.paft_snapshot(),
            )
            self.saver.save_epoch(epoch_schema)
            log_vram_usage(f"{self.method_name}/epoch_{epoch}")

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch} done in {elapsed:.1f}s  "
                f"train_loss={train_loss:.4f}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()
                            if k not in ("train_loss", "epoch"))
            )
            final_metrics = eval_metrics

        # ── final checkpoint ─────────────────────────────────────────────────
        live = self.method.get_live_WV_WO()
        final_schema = FinalSchema(
            metrics          = final_metrics,
            geometric_health = _serialise_health(self.method.geometric_health_snapshot()),
            model_state      = self.method.state_dict()["model"],
            adapted_weights  = {
                "W_V": [t.detach().cpu() for t in live["W_V"]],
                "W_O": [t.detach().cpu() for t in live["W_O"]],
            },
            paft_snapshot    = self.method.paft_snapshot(),
        )
        self.saver.save_final(final_schema)
        log_vram_usage(f"{self.method_name}/final")

        logger.info(f"Training complete.  Run dir: {self.run_dir}")
        return final_metrics

    # ── epoch loops ──────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """One full pass over the training set.  Returns mean train loss."""
        self.method.model.train()
        total_loss   = 0.0
        n_batches    = 0

        for batch in self.train_loader:
            batch = _to_device(batch, self.device)
            loss  = self.method.forward(**batch)

            loss.backward()
            self.method.pre_optimizer_step()   # no-op for all 11 methods

            # Gradient clipping — applied to trainable params only
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.method.model.parameters() if p.requires_grad],
                    max_norm=self.grad_clip,
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss       += loss.item()
            n_batches        += 1
            self._global_step += 1

            if self._global_step % self.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.debug(
                    f"step={self._global_step}  "
                    f"loss={loss.item():.4f}  lr={lr:.2e}"
                )

        return total_loss / max(n_batches, 1)

    def _eval_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate on val_loader.  Returns dict with at minimum 'eval_loss'.
        Additional metrics (accuracy, ROUGE etc.) are injected by the data
        module's collator or by metrics/ functions — the trainer only computes
        loss directly; task metrics are a responsibility of the data layer.
        """
        self.method.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch      = _to_device(batch, self.device)
                loss       = self.method.forward(**batch)
                total_loss += loss.item()
                n_batches  += 1

        return {"eval_loss": total_loss / max(n_batches, 1)}

    # ── init schema builder ──────────────────────────────────────────────────

    def _build_init_schema(self) -> InitSchema:
        """
        Capture pretrained state before any gradient steps.
        Builds decomp_init for PAFT and SVF methods.
        """
        # Geometric health of the pretrained model (reference for all deltas)
        health      = self.method.geometric_health_snapshot()
        decomp_init = None

        _PAFT = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
        _SVF  = {"svf"}

        if self.method_name in _PAFT:
            decomp_init = _build_paft_decomp_init(self.method)
        elif self.method_name in _SVF:
            decomp_init = _build_svf_decomp_init(self.method)

        return InitSchema(
            config           = self.cfg,
            geometric_health = _serialise_health(health),
            decomp_init      = decomp_init,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _serialise_health(health: Dict) -> Dict:
    """
    Convert GeometricHealthMetrics dataclass instances in the health dict
    to plain dicts so they are JSON/torch-serialisable.
    """
    import dataclasses

    def _convert(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(health)


def _build_paft_decomp_init(method: BaseMethod) -> Dict[str, Any]:
    """
    Extract and return the pretrained polar decomposition tensors.
    Called before training — these are the t=0 reference values.
    All tensors CPU.
    """
    snap = method.paft_snapshot()
    if snap is None:
        raise RuntimeError(
            f"paft_snapshot() returned None for method '{method.method_name}'. "
            "PAFT methods must implement paft_snapshot()."
        )

    # Also store the original reconstructed W_V and W_O per head
    live = method.get_live_WV_WO()
    return {
        "W_V_init": [t.detach().cpu() for t in live["W_V"]],
        "W_O_init": [t.detach().cpu() for t in live["W_O"]],
        "Q_V_0":    snap.Q_V,
        "Q_O_0":    snap.Q_O,
        "S_V_0":    snap.S_V,
        "S_O_0":    snap.S_O,
        "EV_V_0":   snap.EV_V,
        "EV_O_0":   snap.EV_O,
        "lam_V_0":  snap.lam_V,
        "lam_O_0":  snap.lam_O,
    }


def _build_svf_decomp_init(method: BaseMethod) -> Dict[str, Any]:
    """Extract pretrained SVD decomposition tensors for SVF method."""
    live = method.get_live_WV_WO()

    # Access SVFModel's stored U, sigma, Vh directly
    svf_model = method.model

    U_V_layers,  sigma_V_layers,  Vh_V_layers  = [], [], []
    U_O_layers,  sigma_O_layers,  Vh_O_layers  = [], [], []

    for _, attn in svf_model.iter_svf_attentions():
        U_V_layers.append(attn.U_V.detach().cpu())
        sigma_V_layers.append(attn.sigma_V.detach().cpu())
        Vh_V_layers.append(attn.Vh_V.detach().cpu())
        U_O_layers.append(attn.U_O.detach().cpu())
        sigma_O_layers.append(attn.sigma_O.detach().cpu())
        Vh_O_layers.append(attn.Vh_O.detach().cpu())

    return {
        "W_V_init":   [t.detach().cpu() for t in live["W_V"]],
        "W_O_init":   [t.detach().cpu() for t in live["W_O"]],
        "U_V_0":      U_V_layers,
        "sigma_V_0":  sigma_V_layers,
        "Vh_V_0":     Vh_V_layers,
        "U_O_0":      U_O_layers,
        "sigma_O_0":  sigma_O_layers,
        "Vh_O_0":     Vh_O_layers,
    }
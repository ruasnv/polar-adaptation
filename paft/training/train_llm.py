"""
train_llm.py — LLM fine-tuning for commonsense reasoning and GSM8K.

Runs one (task, method) pair per invocation.  The shell script calls this
in a loop for 8 commonsense tasks + 1 GSM8K task × 6 methods = 54 runs.

Design matches the existing trainer.py pattern but adds:
  - NF4 quantization handling (no fp16/bf16 autocast on quantized params)
  - LLaMA-specific gradient checkpointing (use_reentrant=False)
  - Log-likelihood evaluation for commonsense (PoLAR-compatible)
  - Generative evaluation for GSM8K (exact-match numeric)
  - Stable rank analysis post-training

Usage:
    python train_llm.py \
        --task boolq \
        --method pure_paft \
        --model_name meta-llama/Llama-3.2-3B \
        --output_dir results/commonsense/boolq/pure_paft \
        --epochs 3 \
        --lr 1e-3 \
        --batch_size 8 \
        --grad_accum 4 \
        --seed 42

    python train_llm.py \
        --task gsm8k \
        --method lora_r8 \
        --output_dir results/gsm8k/lora_r8

For GSM8K --lr 3e-4 and --epochs 3 are recommended (PoLAR convention).

Gradient checkpointing is ALWAYS enabled for LLaMA — reduces activation
memory from ~4 GB to ~2 GB at ~30% extra compute cost.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, set_seed

from paft.methods.llama_methods import (
    get_llama_model, enable_gradient_checkpointing,
    LLAMA_METHODS, DEFAULT_MODEL,
)
from paft.data.commonsense_module import CommonsenseDataModule, SUPPORTED_TASKS as CS_TASKS
from paft.data.gsm8k_module import GSM8KDataModule
from paft.utils.experiment_saver import save_checkpoint, is_complete

logging.basicConfig(
    format  = '%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    level   = logging.INFO,
    datefmt = '%H:%M:%S',
)
logger = logging.getLogger(__name__)

ALL_TASKS = CS_TASKS + ["gsm8k"]


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM fine-tuning for commonsense/GSM8K")

    p.add_argument("--task",       required=True, choices=ALL_TASKS)
    p.add_argument("--method",     required=True, choices=list(LLAMA_METHODS.keys()))
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="results/{task}/{method}")

    # Training hyperparameters
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,   default=8,   help="Micro-batch per GPU")
    p.add_argument("--grad_accum",  type=int,   default=4,   help="Gradient accumulation")
    p.add_argument("--max_length",  type=int,   default=256)
    p.add_argument("--warmup_ratio",type=float, default=0.03)
    p.add_argument("--weight_decay",type=float, default=0.001)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device_map",  default="auto")
    p.add_argument("--log_every",   type=int,   default=50)

    # GSM8K-specific
    p.add_argument("--use_metamath",  action="store_true", default=True)
    p.add_argument("--metamath_size", type=int, default=50_000)
    p.add_argument("--gsm8k_gen_tokens", type=int, default=256)

    # Analysis
    p.add_argument("--run_analysis",  action="store_true", default=True)
    p.add_argument("--skip_analysis", action="store_true", default=False)

    args = p.parse_args()
    args.output_dir = args.output_dir.format(task=args.task, method=args.method)
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

class LLMTrainer:
    """
    Minimal training loop for LLaMA PAFT/LoRA experiments.
    Matches the existing Trainer pattern but stripped to essentials.

    Handles:
      - NF4 model: no fp16 autocast (bitsandbytes manages compute dtype)
      - Gradient checkpointing: enabled always for memory
      - Gradient accumulation: effective_batch = batch_size × grad_accum
      - Log-likelihood evaluation: called at end of each epoch
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        eval_fn,
        args:         argparse.Namespace,
        output_dir:   Path,
    ) -> None:
        self.model        = model
        self.tokenizer    = tokenizer
        self.train_loader = train_loader
        self.eval_fn      = eval_fn
        self.args         = args
        self.output_dir   = output_dir

        # Device — LLaMA with device_map="auto" distributes across available GPUs
        self.device = next(
            (p.device for p in model.parameters() if p.requires_grad),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        logger.info(f"Primary device for trainable params: {self.device}")

        # Optimizer — only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {n_trainable:,}")

        if n_trainable == 0:
            self.optimizer = None
            self.scheduler = None
            logger.info("No trainable params — running evaluation only (frozen baseline).")
            return

        self.optimizer = AdamW(
            trainable_params,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
        )

        # Scheduler: linear warmup + linear decay
        n_steps_per_epoch = len(train_loader)
        n_total_steps     = args.epochs * n_steps_per_epoch // args.grad_accum
        n_warmup_steps    = max(1, int(n_total_steps * args.warmup_ratio))

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps   = n_warmup_steps,
            num_training_steps = n_total_steps,
        )

        logger.info(
            f"Scheduler: {n_warmup_steps} warmup steps, "
            f"{n_total_steps} total steps"
        )

    def train(self) -> Dict[str, Any]:
        """Run all epochs.  Returns dict with final metrics."""
        if self.optimizer is None:
            # Frozen baseline — just evaluate
            logger.info("Frozen baseline — skipping training, evaluating directly.")
            acc = self.eval_fn(self.model)
            return {"accuracy": acc, "epoch": 0, "train_loss": None}

        best_accuracy = 0.0
        history = []

        for epoch in range(self.args.epochs):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            eval_acc   = self.eval_fn(self.model)
            elapsed    = time.time() - t0

            record = {
                "epoch":      epoch,
                "train_loss": train_loss,
                "accuracy":   eval_acc,
            }
            history.append(record)
            best_accuracy = max(best_accuracy, eval_acc)

            logger.info(
                f"Epoch {epoch}/{self.args.epochs - 1}  "
                f"loss={train_loss:.4f}  acc={eval_acc:.4f}  "
                f"elapsed={elapsed:.0f}s"
            )

            # Save epoch checkpoint
            epoch_dir = self.output_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            _save_trainable_params(self.model, epoch_dir / "adapter.pt")
            with open(epoch_dir / "metrics.json", "w") as f:
                json.dump(record, f, indent=2)
            # Optimizer + scheduler state (for resume)
            torch.save(self.optimizer.state_dict(), epoch_dir / "optimizer.pt")
            torch.save(self.scheduler.state_dict(), epoch_dir / "scheduler.pt")
            # PAFT snapshot + geometric health per epoch
            save_checkpoint(
                model       = self.model,
                output_dir  = self.output_dir,
                tag         = f"epoch_{epoch}",
                method_name = self.args.method,
                metrics     = record,
                model_type  = "llama",
            )

        # Save training history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return {
            "accuracy":   best_accuracy,
            "final_loss": history[-1]["train_loss"] if history else None,
        }

    def _train_epoch(self, epoch: int) -> float:
        """One full pass over training data.  Returns mean loss."""
        self.model.train()
        total_loss      = 0.0
        n_micro_batches = 0
        accum_loss      = 0.0
        global_step     = 0

        self.optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(self.train_loader):
            # Move non-quantized tensors to the primary device
            batch = _to_device(batch, self.device)

            # Forward pass — bitsandbytes manages compute dtype internally
            # Do NOT use torch.autocast here; it conflicts with NF4 compute type
            outputs = self.model(**batch)
            loss    = outputs.loss / self.args.grad_accum
            loss.backward()

            accum_loss      += loss.item() * self.args.grad_accum
            total_loss      += loss.item() * self.args.grad_accum
            n_micro_batches += 1

            is_accum_step = ((micro_step + 1) % self.args.grad_accum == 0)
            is_last_batch = (micro_step + 1 == len(self.train_loader))

            if is_accum_step or is_last_batch:
                # Gradient clipping on trainable params only
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm = self.args.grad_clip,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % self.args.log_every == 0:
                    avg_loss = accum_loss / self.args.grad_accum
                    lr_now   = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  epoch={epoch}  step={global_step}  "
                        f"loss={avg_loss:.4f}  lr={lr_now:.2e}"
                    )
                accum_loss = 0.0

        return total_loss / max(n_micro_batches, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task: {args.task}  Method: {args.method}  Output: {output_dir}")
    logger.info(f"Model: {args.model_name}")

    # Resume: skip completed runs
    if is_complete(output_dir):
        logger.info("Run already complete (training_complete found). Skipping.")
        return

    # ── 1. Build model ───────────────────────────────────────────────────────
    logger.info(f"Building {args.method} ...")
    model, tokenizer = get_llama_model(args.method, args.model_name, args.device_map)

    # Gradient checkpointing — essential for 8 GB VRAM
    if args.method != "frozen":
        enable_gradient_checkpointing(model)

    # ── 2. Data ──────────────────────────────────────────────────────────────
    is_gsm8k = (args.task == "gsm8k")

    if is_gsm8k:
        dm = GSM8KDataModule(
            tokenizer       = tokenizer,
            use_metamath    = args.use_metamath,
            metamath_subset = args.metamath_size,
            max_length      = args.max_length,
            batch_size      = args.batch_size,
        )
        train_loader = dm.get_train_loader()

        # Evaluation function: greedy generation exact match
        device = next(
            (p.device for p in model.parameters() if p.requires_grad),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        def eval_fn(m):
            return dm.evaluate_generation(
                m, device,
                max_new_tokens = args.gsm8k_gen_tokens,
                n_examples     = 200,   # quick eval during training; full eval at end
            )

    else:
        dm = CommonsenseDataModule(
            task_name       = args.task,
            tokenizer       = tokenizer,
            max_length      = args.max_length,
            batch_size      = args.batch_size,
        )
        train_loader = dm.get_train_loader()

        # Evaluation function: log-likelihood multiple choice (PoLAR protocol)
        device = next(
            (p.device for p in model.parameters() if p.requires_grad),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        def eval_fn(m):
            return dm.evaluate_log_likelihood(m, device)

    # ── 3. Save INIT checkpoint ────────────────────────────────────────────────
    logger.info("Saving init checkpoint ...")
    save_checkpoint(
        model       = model,
        output_dir  = output_dir,
        tag         = "init",
        method_name = args.method,
        metrics     = {"task": args.task, "method": args.method, "stage": "init"},
        model_type  = "llama",
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    trainer = LLMTrainer(
        model        = model,
        tokenizer    = tokenizer,
        train_loader = train_loader,
        eval_fn      = eval_fn,
        args         = args,
        output_dir   = output_dir,
    )

    final_metrics = trainer.train()

    # ── 4. Final evaluation (full dataset) ────────────────────────────────────
    logger.info("Running final full evaluation ...")
    model.eval()
    if is_gsm8k:
        final_acc = dm.evaluate_generation(
            model, device,
            max_new_tokens = args.gsm8k_gen_tokens,
        )
    else:
        final_acc = dm.evaluate_log_likelihood(model, device)

    final_metrics["final_accuracy"] = final_acc
    final_metrics["task"]   = args.task
    final_metrics["method"] = args.method
    final_metrics["trainable_params"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info(f"Final accuracy on {args.task}: {final_acc:.4f}")

    # ── 5. Save results ───────────────────────────────────────────────────────
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save final adapter weights (fast, small)
    _save_trainable_params(model, output_dir / "adapter_final.pt")

    # ── Save FINAL checkpoint (adapted weights + PAFT snapshot + sentinel) ───
    logger.info("Saving final checkpoint ...")
    save_checkpoint(
        model       = model,
        output_dir  = output_dir,
        tag         = "final",
        method_name = args.method,
        metrics     = final_metrics,
        model_type  = "llama",
    )

    # ── 6. Geometric analysis ─────────────────────────────────────────────────
    if args.run_analysis and not args.skip_analysis and args.method in ("pure_paft", "hybrid_paft"):
        _run_llama_analysis(model, args.method, output_dir)

    logger.info(f"\nDone!  Results saved to {output_dir}")
    logger.info(f"Final accuracy: {final_acc:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def _run_llama_analysis(model, method_name: str, output_dir: Path) -> None:
    """Stable rank analysis on LLaMA PAFT v_proj weights."""
    from paft.model.llama_paft_model import LLaMAPAFTModel
    from paft.analysis.stable_rank import analyze_all_layers, summarize_stable_rank

    if not isinstance(model, LLaMAPAFTModel):
        return

    logger.info("Running geometric analysis on LLaMA PAFT v_proj ...")
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # get_live_WV_WO returns W_V [H_kv, n_embd, head_dim] per layer
    live = model.get_live_WV_WO()
    W_V_layers = live["W_V"]

    # Flatten [H_kv, n_embd, d] → [H_kv * n_embd, d] per layer for stable rank
    # (equivalent to treating the full v_proj weight as a single matrix)
    W_V_2d = [
        W.reshape(-1, W.shape[-1])   # [H_kv * n_embd, d_head]
        for W in W_V_layers
    ]
    layer_metrics = analyze_all_layers(W_V_2d)
    summary = summarize_stable_rank(W_V_2d)

    analysis = {
        "method":      method_name,
        "summary":     summary,
        "ortho_error": model.measure_orthogonality(),
        "note":        "Metrics computed on v_proj only (PAFT target). "
                       "o_proj is frozen NF4 (same for all methods).",
    }
    with open(analysis_dir / "stable_rank.json", "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"sr(W_eff) mean: {summary.get('stable_rank_Weff', 'N/A'):.3f}")
    logger.info(f"Orthogonality error: {analysis['ortho_error']:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _to_device(batch: Dict, device: torch.device) -> Dict:
    """Move tensor values in batch to device.  Skip non-tensors."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _save_trainable_params(model: nn.Module, path: Path) -> None:
    """Save only trainable parameters — much smaller than full model state."""
    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if state:
        torch.save(state, path)
        size_mb = path.stat().st_size / 1e6
        logger.info(f"Saved {len(state)} adapter tensors ({size_mb:.1f} MB) → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
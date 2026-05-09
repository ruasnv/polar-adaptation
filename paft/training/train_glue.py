"""
train_glue.py — GLUE benchmark training for DeBERTa-v3-base.

Runs one (task, method) pair per invocation.  The orchestration shell script
calls this in a loop for all 8 tasks × 8 methods = 64 runs.

HuggingFace Trainer handles: mixed precision, gradient accumulation,
eval loop, metric computation, logging, and checkpointing.  This keeps
the GLUE training correct and directly comparable to PoLAR / LoRA papers
which also use the HF Trainer for GLUE.

Usage:
    python train_glue.py \
        --task cola \
        --method pure_paft \
        --output_dir results/glue/cola/pure_paft \
        --epochs 5 \
        --lr 2e-4 \
        --batch_size 32 \
        --grad_accum 1 \
        --seed 42

After training, runs stable rank analysis and saves to {output_dir}/analysis/.

Output directory structure:
    {output_dir}/
        hf_checkpoints/      HuggingFace Trainer checkpoints
        analysis/
            stable_rank.json     sr(W_eff), sr(ΔW), directional diversity
            layer_profile.pt     per-layer metrics
            figures/
                stable_rank.pdf
                layer_profile.pdf
        metrics.json             final GLUE metrics
        config.json              full training config
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

from paft.data.glue_module import GLUEDataModule, NUM_LABELS, SUPPORTED_TASKS
from paft.methods.deberta_methods import get_deberta_model, DEBERTA_METHODS
from analysis.stable_rank import (
    analyze_all_layers, compare_methods_stable_rank,
    plot_stable_rank_comparison, plot_layer_profile,
)

logging.basicConfig(
    format  = '%(asctime)s  %(levelname)-8s  %(message)s',
    level   = logging.INFO,
    datefmt = '%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GLUE fine-tuning for DeBERTa-v3-base")

    # Task and method
    p.add_argument("--task",   required=True, choices=SUPPORTED_TASKS)
    p.add_argument("--method", required=True, choices=list(DEBERTA_METHODS.keys()))

    # Paths
    p.add_argument("--output_dir",  default="results/glue/{task}/{method}")

    # Training hyperparameters (matched to PoLAR / DeBERTa paper recommendations)
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch_size",  type=int,   default=32,  help="Per-device train batch")
    p.add_argument("--grad_accum",  type=int,   default=1)
    p.add_argument("--max_length",  type=int,   default=512)
    p.add_argument("--warmup_ratio",type=float, default=0.06)
    p.add_argument("--weight_decay",type=float, default=0.01)

    # System
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--fp16",       action="store_true", default=True)
    p.add_argument("--no_fp16",    action="store_true", default=False,
                   help="Disable fp16 (useful for debugging)")

    # Analysis
    p.add_argument("--run_analysis", action="store_true", default=True,
                   help="Run stable rank analysis after training")
    p.add_argument("--skip_analysis", action="store_true", default=False)

    args = p.parse_args()

    # Expand {task} and {method} in output_dir
    args.output_dir = args.output_dir.format(task=args.task, method=args.method)
    args.fp16 = args.fp16 and not args.no_fp16

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Training entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis" / "figures").mkdir(parents=True, exist_ok=True)

    logger.info(f"Task: {args.task}  Method: {args.method}  Output: {output_dir}")

    # ── 1. Build model + tokenizer ───────────────────────────────────────────
    num_labels = NUM_LABELS[args.task]
    logger.info(f"Building {args.method} for {args.task} (num_labels={num_labels}) ...")
    model, tokenizer = get_deberta_model(args.method, num_labels)
    logger.info(
        f"Model ready — "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        "trainable params"
    )

    # ── 2. Data ──────────────────────────────────────────────────────────────
    dm = GLUEDataModule(
        task_name  = args.task,
        tokenizer  = tokenizer,
        max_length = args.max_length,
        batch_size = args.batch_size,
    )
    logger.info(dm)

    # ── 3. HuggingFace TrainingArguments ─────────────────────────────────────
    # Task-specific hyperparameters (following DeBERTa-v3 paper recommendations)
    # Epoch counts: more for small tasks (RTE: 10, CoLA: 10), fewer for large (MNLI: 3, QQP: 3)
    TASK_EPOCHS = {
        "cola": 10, "mnli": 3,  "mrpc": 10, "qnli": 5,
        "qqp":  3,  "rte":  10, "sst2": 5,  "stsb": 10,
    }
    TASK_LR = {
        "cola": 2e-5, "mnli": 1.5e-5, "mrpc": 2e-5, "qnli": 2e-5,
        "qqp":  1e-5, "rte":  2e-5,   "sst2": 2e-5, "stsb": 2e-5,
    }
    # Use command-line lr if explicitly set, otherwise use task defaults
    effective_lr     = args.lr if args.lr != 2e-4 else TASK_LR.get(args.task, args.lr)
    effective_epochs = args.epochs if args.epochs != 5 else TASK_EPOCHS.get(args.task, args.epochs)

    # For PAFT methods, use a higher learning rate on the S/lam parameters
    if args.method in ("pure_paft", "hybrid_paft"):
        effective_lr = max(effective_lr, 1e-3)

    training_args = TrainingArguments(
        output_dir                   = str(output_dir / "hf_checkpoints"),
        num_train_epochs             = effective_epochs,
        per_device_train_batch_size  = args.batch_size,
        per_device_eval_batch_size   = args.batch_size * 2,
        gradient_accumulation_steps  = args.grad_accum,
        learning_rate                = effective_lr,
        weight_decay                 = args.weight_decay,
        warmup_ratio                 = args.warmup_ratio,
        lr_scheduler_type            = "linear",
        fp16                         = args.fp16 and torch.cuda.is_available(),
        eval_strategy                = "epoch",
        save_strategy                = "epoch",
        load_best_model_at_end       = True,
        metric_for_best_model        = dm.primary_metric,
        greater_is_better            = True,
        seed                         = args.seed,
        logging_steps                = 50,
        report_to                    = ["none"],   # no wandb/tensorboard unless desired
        label_names                  = ["labels"],
        remove_unused_columns        = False,
        dataloader_num_workers       = 4,
    )

    # ── 4. Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = dm._train_dataset,
        eval_dataset     = dm._val_dataset,
        compute_metrics  = dm.get_metric_fn(),
        tokenizer        = tokenizer,
    )

    logger.info("Starting training ...")
    trainer.train()

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    logger.info("Final evaluation ...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    # Clean up metric names (remove 'eval_' prefix)
    metrics = {k.replace("eval_", ""): v for k, v in eval_results.items()}
    metrics["task"]   = args.task
    metrics["method"] = args.method
    metrics["trainable_params"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config
    config = vars(args)
    config["effective_lr"] = effective_lr
    config["effective_epochs"] = effective_epochs
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── 6. Geometric analysis ─────────────────────────────────────────────────
    if args.run_analysis and not args.skip_analysis:
        _run_geometric_analysis(model, args.method, output_dir)

    logger.info(f"\nDone!  Results saved to {output_dir}")
    logger.info(f"Primary metric ({dm.primary_metric}): {metrics.get(dm.primary_metric, 'N/A'):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Geometric analysis (runs post-training, no VRAM needed)
# ──────────────────────────────────────────────────────────────────────────────

def _run_geometric_analysis(model, method_name: str, output_dir: Path) -> None:
    """
    Run stable rank and geometric health analysis on the trained model.
    Saves results to {output_dir}/analysis/.
    """
    from paft.model.deberta_paft_model import DeBERTaPAFTModel
    from analysis.stable_rank import analyze_all_layers, summarize_stable_rank

    analysis_dir = output_dir / "analysis"

    logger.info("Running geometric analysis ...")
    model.eval()

    # Get live weight matrices
    if isinstance(model, DeBERTaPAFTModel):
        live_weights = model.get_live_WV_WO()
        W_V_layers   = live_weights["W_V"]  # [H, d, n] per layer
        W_O_layers   = live_weights["W_O"]  # [H, n, d] per layer

        # W_V: [H, n_embd, d_head] per layer — reshape to [H*n_embd, d_head] for stable rank
        # This treats the full value_proj matrix as a single [n_embd, d_head]-equivalent
        # weight (averaged view), consistent with how o_proj is treated in the paper.
        W_V_2d = [W.reshape(-1, W.shape[-1]) for W in W_V_layers]   # [H*n, d] per layer
        W_O_2d = [W.reshape(W.shape[0], -1) for W in W_O_layers]    # [H, d*n] or reshape to [d, H*n]

        # Compute stable rank for V and O projections
        sr_V = analyze_all_layers(W_V_2d)
        sr_O = analyze_all_layers(W_O_2d)

        stable_rank_results = {
            "method":          method_name,
            "V_projection":    {k: float(sum(v)/len(v)) for k, v in sr_V.items()},
            "O_projection":    {k: float(sum(v)/len(v)) for k, v in sr_O.items()},
        }
        # Also measure orthogonality if PAFT model
        ortho = model.measure_orthogonality()
        stable_rank_results["orthogonality"] = ortho

    else:
        # Non-PAFT model — just report basic metrics
        stable_rank_results = {
            "method":  method_name,
            "note":    "Non-PAFT model — per-head W_eff not directly accessible",
        }

    with open(analysis_dir / "stable_rank.json", "w") as f:
        json.dump(stable_rank_results, f, indent=2)

    logger.info(f"Geometric analysis saved to {analysis_dir}")
    if "V_projection" in stable_rank_results:
        sr_val = stable_rank_results["V_projection"].get("stable_rank_Weff", "N/A")
        logger.info(f"  Mean sr(W_eff) for V projection: {sr_val:.3f}" if isinstance(sr_val, float) else f"  sr(W_eff): {sr_val}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
"""
GLUE data module for DeBERTa-v3-base experiments.

Handles all 8 GLUE tasks used in LoRA and PoLAR papers:
  CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B

Skips WNLI (adversarial, routinely excluded by PEFT papers).

Each task returns a HuggingFace Dataset pre-tokenised to max_length=512.
The module also provides the correct metric function (from 'evaluate' library)
and num_labels for the classification head.

STS-B is a regression task (num_labels=1, MSE loss, Pearson/Spearman metric).
All others are classification (CrossEntropy loss, accuracy / MCC / F1).

Usage in train_glue.py:
    dm = GLUEDataModule("cola", tokenizer, max_length=512, batch_size=32)
    train_loader = dm.get_train_loader()
    val_loader   = dm.get_val_loader()
    metric_fn    = dm.get_metric_fn()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# ── Task configuration ────────────────────────────────────────────────────────

# Maps task name → (sentence1_key, sentence2_key | None)
TASK_TO_KEYS: Dict[str, tuple] = {
    "cola":  ("sentence", None),
    "mnli":  ("premise", "hypothesis"),
    "mrpc":  ("sentence1", "sentence2"),
    "qnli":  ("question", "sentence"),
    "qqp":   ("question1", "question2"),
    "rte":   ("sentence1", "sentence2"),
    "sst2":  ("sentence", None),
    "stsb":  ("sentence1", "sentence2"),
}

# Maps task name → primary metric name (from 'evaluate' library)
TASK_TO_METRIC: Dict[str, str] = {
    "cola":  "matthews_correlation",
    "mnli":  "accuracy",
    "mrpc":  "f1",
    "qnli":  "accuracy",
    "qqp":   "f1",
    "rte":   "accuracy",
    "sst2":  "accuracy",
    "stsb":  "pearson",
}

# Maps task name → number of output labels
NUM_LABELS: Dict[str, int] = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp":  2,
    "rte":  2,
    "sst2": 2,
    "stsb": 1,   # regression
}

# Validation split name (MNLI has matched/mismatched)
VALIDATION_KEY: Dict[str, str] = {
    "cola": "validation",
    "mnli": "validation_matched",
    "mrpc": "validation",
    "qnli": "validation",
    "qqp":  "validation",
    "rte":  "validation",
    "sst2": "validation",
    "stsb": "validation",
}

SUPPORTED_TASKS = list(TASK_TO_KEYS.keys())


class GLUEDataModule:
    """
    Pre-tokenises a GLUE task and provides DataLoaders.

    Args:
        task_name:   One of SUPPORTED_TASKS.
        tokenizer:   DeBERTa-v3 tokenizer from AutoTokenizer.
        max_length:  Truncation length.  512 is standard for DeBERTa.
        batch_size:  Micro-batch size for train DataLoader.
        val_batch_size: Batch size for validation (can be 2× train for speed).
        num_workers: DataLoader worker processes.
    """

    def __init__(
        self,
        task_name:      str,
        tokenizer:      PreTrainedTokenizer,
        max_length:     int = 512,
        batch_size:     int = 32,
        val_batch_size: int = 64,
        num_workers:    int = 4,
    ) -> None:
        if task_name not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task_name}'. "
                f"Choose from: {SUPPORTED_TASKS}"
            )

        self.task_name      = task_name
        self.tokenizer      = tokenizer
        self.max_length     = max_length
        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers    = num_workers
        self.num_labels     = NUM_LABELS[task_name]
        self.is_regression  = (self.num_labels == 1)

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

        logger.info(f"Loading GLUE/{task_name} ...")
        raw = load_dataset("glue", task_name)

        self._train_dataset = self._tokenise(raw["train"])
        self._val_dataset   = self._tokenise(raw[VALIDATION_KEY[task_name]])

        logger.info(
            f"GLUE/{task_name}: "
            f"train={len(self._train_dataset)}  "
            f"val={len(self._val_dataset)}  "
            f"num_labels={self.num_labels}"
        )

    # ── tokenisation ──────────────────────────────────────────────────────────

    def _tokenise(self, dataset) -> Any:
        """Tokenise and format a HF dataset split for DeBERTa classification."""
        key1, key2 = TASK_TO_KEYS[self.task_name]

        def _encode(examples):
            args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
            result = self.tokenizer(
                *args,
                truncation   = True,
                max_length   = self.max_length,
                padding      = 'max_length',
                return_tensors = None,  # return lists; DataCollator will tensorise
            )
            # HF datasets: label → labels (HF Trainer convention)
            result["labels"] = examples["label"]
            return result

        tokenised = dataset.map(
            _encode,
            batched        = True,
            remove_columns = [c for c in dataset.column_names if c != "label"],
            desc           = f"Tokenising {self.task_name}",
        )
        tokenised.set_format("torch")
        return tokenised

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size  = self.val_batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    # ── metric ────────────────────────────────────────────────────────────────

    def get_metric_fn(self) -> Callable:
        """
        Return a compute_metrics function compatible with HF Trainer.
        Accepts (logits, labels) and returns {metric_name: value}.
        """
        try:
            import evaluate
        except ImportError:
            raise ImportError("evaluate library required: pip install evaluate")

        metric     = evaluate.load("glue", self.task_name)
        task_name  = self.task_name
        is_reg     = self.is_regression

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            if is_reg:
                preds = logits.squeeze()
            else:
                preds = logits.argmax(axis=-1)
            result = metric.compute(predictions=preds, references=labels)
            return result

        return compute_metrics

    # ── convenience ───────────────────────────────────────────────────────────

    def num_train_samples(self) -> int:
        return len(self._train_dataset)

    def num_val_samples(self) -> int:
        return len(self._val_dataset)

    @property
    def primary_metric(self) -> str:
        return TASK_TO_METRIC[self.task_name]

    def __repr__(self) -> str:
        return (
            f"GLUEDataModule(task={self.task_name!r}, "
            f"train={len(self._train_dataset)}, "
            f"val={len(self._val_dataset)}, "
            f"num_labels={self.num_labels})"
        )
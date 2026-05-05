"""
News domain data module (small domain shift).

Tasks
─────
1. Summarisation  — CNN/DailyMail 3.0.0
   Format: "Article: {article}\nSummary: {highlights}"
   Eval metric: ROUGE-1/2/L

2. Sentiment      — SST-2 (GLUE)
   Format: "Review: {sentence}\nSentiment: {positive|negative}"
   Eval metric: accuracy

For training, both tasks are formatted as causal LM sequences.
GPT-2 is trained to predict the full sequence including the label/summary.
Task-specific eval metrics are computed post-hoc by metrics/generation.py
and metrics/classification.py, not during training.

Dataset sizes (approximate):
    CNN/DailyMail train: 287K   val: 13.4K
    SST-2 train: 67K            val: 872

We cap samples at _MAX_SAMPLES to keep epoch times manageable.
"""

from __future__ import annotations

import logging
from typing import List

from datasets import load_dataset

from paft.data.base import BaseDataModule

logger = logging.getLogger(__name__)

# Cap samples per split to keep run times manageable across 100 experiments.
# CNN/DM at 20K samples: ~40 min/epoch on GPT-2 small; full set ~4 h/epoch.
_MAX_TRAIN = 20_000
_MAX_VAL   = 2_000


class NewsDataModule(BaseDataModule):
    """
    Combined news domain: CNN/DailyMail (summarisation) + SST-2 (sentiment).

    The two tasks are interleaved in the training set so the model sees both
    within each epoch.  Eval loaders are separate per task but combined into
    one loader here for the trainer's single eval pass.

    task: "summarization" | "sentiment" | "combined" (default)
    """

    def __init__(self, cfg: dict, hf_name: str, task: str = "combined") -> None:
        self.task = task
        super().__init__(cfg, hf_name)

    def _load_train_texts(self) -> List[str]:
        texts = []
        if self.task in ("summarization", "combined"):
            texts += _load_cnn_dm(split="train", max_samples=_MAX_TRAIN)
        if self.task in ("sentiment", "combined"):
            texts += _load_sst2(split="train", max_samples=_MAX_TRAIN // 4)
        return texts

    def _load_val_texts(self) -> List[str]:
        texts = []
        if self.task in ("summarization", "combined"):
            texts += _load_cnn_dm(split="validation", max_samples=_MAX_VAL)
        if self.task in ("sentiment", "combined"):
            texts += _load_sst2(split="validation", max_samples=_MAX_VAL // 4)
        return texts


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_cnn_dm(split: str, max_samples: int) -> List[str]:
    """Load CNN/DailyMail and format as article → summary."""
    logger.info(f"Loading CNN/DailyMail ({split}, max={max_samples}) ...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
    texts = []
    for item in ds.select(range(min(max_samples, len(ds)))):
        article  = item["article"].strip().replace("\n", " ")
        summary  = item["highlights"].strip().replace("\n", " ")
        # Truncate article to leave room for the summary in the 512-token window
        article  = " ".join(article.split()[:300])
        texts.append(f"Article: {article}\nSummary: {summary}")
    return texts


def _load_sst2(split: str, max_samples: int) -> List[str]:
    """Load SST-2 and format as sentence → sentiment label."""
    logger.info(f"Loading SST-2 ({split}, max={max_samples}) ...")
    ds = load_dataset("glue", "sst2", split=split)
    label_map = {0: "negative", 1: "positive"}
    texts = []
    for item in ds.select(range(min(max_samples, len(ds)))):
        sentence = item["sentence"].strip()
        label    = label_map[item["label"]]
        texts.append(f"Review: {sentence}\nSentiment: {label}")
    return texts
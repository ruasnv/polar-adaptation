"""
Perplexity computation on held-out token sequences.

Perplexity is the primary training-time signal — computed on the val loader
by the trainer (via eval_loss → exp(eval_loss)).  This module provides
higher-level helpers used in post-hoc analysis:

1. dataset_perplexity  — PPL on a raw text dataset (no DataLoader needed)
2. domain_perplexity   — PPL across a whole domain data module

Perplexity is used by domain_correlation.py to measure distribution shift:
the ratio of fine-tuned PPL to pretrained PPL on a domain measures how much
that domain differs from GPT-2's pretraining distribution.
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from paft.data.utils import tokenise_and_label, ListDataset, build_dataloader

logger = logging.getLogger(__name__)


def eval_loss_to_ppl(eval_loss: float) -> float:
    """Convert mean cross-entropy loss to perplexity.  ppl = exp(loss)."""
    return math.exp(eval_loss)


def dataset_perplexity(
    model,
    tokenizer:  PreTrainedTokenizer,
    texts:      List[str],
    max_length: int         = 512,
    batch_size: int         = 16,
    device:     torch.device = torch.device("cpu"),
) -> float:
    """
    Compute perplexity of `model` on a list of text strings.

    Tokenises texts, runs forward passes in batches (no_grad), returns
    exp(mean cross-entropy loss) across all tokens.

    Args:
        model:      Any model that accepts (input_ids, attention_mask, labels)
                    and returns an object with a .loss attribute.
        tokenizer:  Matching tokeniser.
        texts:      List of text strings to evaluate on.
        max_length: Token sequence length for truncation/padding.
        batch_size: Eval batch size.
        device:     Evaluation device.

    Returns:
        Perplexity as a float.  Lower is better.
    """
    model.eval()
    samples  = [tokenise_and_label(t, tokenizer, max_length) for t in texts]
    dataset  = ListDataset(samples)
    loader   = build_dataloader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            total_loss += out.loss.item()
            n_batches  += 1

    mean_loss = total_loss / max(n_batches, 1)
    return eval_loss_to_ppl(mean_loss)


def domain_perplexity(
    model,
    val_loader: DataLoader,
    device:     torch.device = torch.device("cpu"),
) -> float:
    """
    Compute perplexity over a pre-built DataLoader.

    Used by domain_correlation.py after loading a checkpoint.

    Returns:
        Perplexity as a float.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out   = model(**batch)
            total_loss += out.loss.item()
            n_batches  += 1

    mean_loss = total_loss / max(n_batches, 1)
    return eval_loss_to_ppl(mean_loss)
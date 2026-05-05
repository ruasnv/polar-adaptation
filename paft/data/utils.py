"""
Shared tokenisation helpers and DataLoader construction.

All data modules use get_tokenizer() and build_dataloader() from here.
This centralises the tokeniser configuration so any change (e.g. padding
side, truncation strategy) is made once and propagates everywhere.

GPT-2 tokeniser notes
─────────────────────
GPT-2 has no pad token by default.  We set pad_token = eos_token, which is
the standard workaround.  Padding is on the right so that attention masks
correctly indicate valid tokens from the left.

Label convention for causal LM
────────────────────────────────
We train with labels = input_ids.  GPT-2LMHeadModel shifts labels
internally (predicts token i+1 from token i), so passing the same tensor
for both input_ids and labels is correct.  Padding positions must receive
label = -100 so they are ignored by cross-entropy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Padding label ignored by cross-entropy
IGNORE_INDEX = -100


def get_tokenizer(hf_name: str) -> PreTrainedTokenizer:
    """
    Load the tokeniser for a HuggingFace model and apply GPT-2 workarounds.

    Sets pad_token = eos_token and padding_side = "right".
    Safe to call multiple times — tokenisers are lightweight to load.
    """
    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


def tokenise_and_label(
    text:       str,
    tokenizer:  PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise a single text string for causal LM training.

    Returns:
        input_ids      [max_length]  — token ids, right-padded with pad_token_id
        attention_mask [max_length]  — 1 for real tokens, 0 for padding
        labels         [max_length]  — same as input_ids but padding → IGNORE_INDEX

    Truncation: right-truncate at max_length.
    """
    enc = tokenizer(
        text,
        max_length  = max_length,
        truncation  = True,
        padding     = "max_length",
        return_tensors = "pt",
    )
    input_ids      = enc["input_ids"].squeeze(0)       # [T]
    attention_mask = enc["attention_mask"].squeeze(0)   # [T]

    # Mask padding positions in labels
    labels = input_ids.clone()
    labels[attention_mask == 0] = IGNORE_INDEX

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def build_dataloader(
    dataset:    Dataset,
    batch_size: int,
    shuffle:    bool,
    num_workers: int = 0,
) -> DataLoader:
    """
    Wrap a Dataset in a DataLoader with standard settings.

    num_workers=0 by default — HuggingFace datasets already do internal
    caching; extra worker processes can cause tokeniser fork issues.
    Increase to 2–4 for large datasets if CPU is the bottleneck.
    """
    return DataLoader(
        dataset,
        batch_size   = batch_size,
        shuffle      = shuffle,
        num_workers  = num_workers,
        pin_memory   = torch.cuda.is_available(),
        drop_last    = shuffle,   # drop last incomplete batch during training only
    )


class ListDataset(Dataset):
    """
    Simple Dataset wrapping a list of pre-tokenised sample dicts.

    Each item is a dict with keys: input_ids, attention_mask, labels.
    All values are tensors of shape [max_length].
    """

    def __init__(self, samples: List[Dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
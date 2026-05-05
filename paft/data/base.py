"""
BaseDataModule — abstract contract for all domain data modules.

Each domain module (news, legal, biomedical, code) subclasses this and
implements _load_and_format(), which returns a list of text strings.
BaseDataModule handles tokenisation and DataLoader construction uniformly.

Design decisions
────────────────
1. Text formatting is the domain module's responsibility.
   Each domain formats its dataset as a single string per sample:
       "<task_prefix>: {input}\n<target_prefix>: {target}"
   GPT-2 is trained to predict the full sequence, including the target.

2. Tokenisation is done eagerly in __init__, not lazily.
   This pays the tokenisation cost once at startup rather than per-batch,
   which matters when the same dataset is used across multiple epochs.
   For GPT-2 small (max_seq_len=512) and typical dataset sizes, the full
   tokenised dataset fits in CPU RAM.

3. The trainer receives DataLoaders that yield:
       {input_ids [B,T], attention_mask [B,T], labels [B,T]}
   The trainer calls method.forward(**batch) which matches BaseMethod.forward's
   signature exactly.

4. Eval split: use the official val/dev split if available.
   For datasets without a val split, take the last 10% of the training set.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from paft.data.utils import (
    get_tokenizer,
    tokenise_and_label,
    build_dataloader,
    ListDataset,
)

logger = logging.getLogger(__name__)


class BaseDataModule(ABC):
    """
    Abstract data module.

    Subclass and implement:
        _load_train_texts() -> List[str]   — formatted training samples
        _load_val_texts()   -> List[str]   — formatted validation samples

    The returned strings are the complete model input including the target,
    e.g. "Article: ...\nSummary: ..." or "Review: ...\nSentiment: positive".
    """

    def __init__(
        self,
        cfg:        dict,
        hf_name:    str,
        max_length: int | None = None,
    ) -> None:
        """
        Args:
            cfg:        Full merged experiment config.
            hf_name:    HuggingFace model name, e.g. "gpt2".  Used to load
                        the matching tokeniser.
            max_length: Token sequence length.  Defaults to cfg training value.
        """
        self.cfg        = cfg
        self.hf_name    = hf_name
        self.max_length = max_length or cfg["training"]["max_seq_len"]
        self.batch_size = cfg["training"]["batch_size"]

        self.tokenizer: PreTrainedTokenizer = get_tokenizer(hf_name)

        logger.info(
            f"[{self.__class__.__name__}] Loading dataset  "
            f"max_length={self.max_length}  batch_size={self.batch_size}"
        )

        train_texts = self._load_train_texts()
        val_texts   = self._load_val_texts()

        logger.info(
            f"[{self.__class__.__name__}] "
            f"train={len(train_texts)} samples  val={len(val_texts)} samples"
        )

        self._train_dataset = self._tokenise(train_texts)
        self._val_dataset   = self._tokenise(val_texts)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_train_texts(self) -> List[str]:
        """Return formatted training strings (full input + target)."""

    @abstractmethod
    def _load_val_texts(self) -> List[str]:
        """Return formatted validation strings."""

    # ------------------------------------------------------------------
    # Public API (called by the trainer / scripts)
    # ------------------------------------------------------------------

    def get_train_loader(self) -> DataLoader:
        """DataLoader with shuffle=True for training."""
        return build_dataloader(
            self._train_dataset,
            batch_size = self.batch_size,
            shuffle    = True,
        )

    def get_val_loader(self) -> DataLoader:
        """DataLoader with shuffle=False for evaluation."""
        return build_dataloader(
            self._val_dataset,
            batch_size = self.batch_size,
            shuffle    = False,
        )

    def num_train_samples(self) -> int:
        return len(self._train_dataset)

    def num_val_samples(self) -> int:
        return len(self._val_dataset)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenise(self, texts: List[str]) -> ListDataset:
        """Tokenise a list of text strings into a ListDataset."""
        samples = [
            tokenise_and_label(text, self.tokenizer, self.max_length)
            for text in texts
        ]
        return ListDataset(samples)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _holdout_split(
        texts: List[str],
        val_fraction: float = 0.1,
    ) -> Tuple[List[str], List[str]]:
        """
        Split a list of texts into train/val.
        Used when a dataset has no official val split.
        Takes the last val_fraction as val (not shuffled — preserves order).
        """
        n_val = max(1, int(len(texts) * val_fraction))
        return texts[:-n_val], texts[-n_val:]
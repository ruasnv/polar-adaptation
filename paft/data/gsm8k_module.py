"""
GSM8K mathematical reasoning data module for LLaMA experiments.

Training:  Fine-tune on MetaMathQA (50K diverse rewrites of GSM8K + MATH)
           OR directly on GSM8K train split (7.5K examples).
           MetaMathQA is preferred — it augments the training data and yields
           higher final accuracy (same approach as PoLAR's math experiments).

Evaluation: Exact match on the numeric final answer.
            Extract number from model's generation after "The answer is:"

Answer extraction:
    The model is trained to produce: "... The answer is {number}."
    We extract the LAST number appearing in the generation.
    Non-numeric answers count as wrong (not scored as partial).

Metric: Accuracy = correct_numeric_matches / total_examples
        This is the standard GSM8K metric (Cobbe et al. 2021).

Usage:
    dm = GSM8KDataModule(tokenizer, use_metamath=True, batch_size=4)
    train_loader = dm.get_train_loader()
    accuracy = dm.evaluate_generation(model, device, max_new_tokens=256)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Answer extraction pattern — match the last number (possibly decimal) in output
_NUMBER_PATTERN = re.compile(r'[\-]?\d+[\.,]?\d*')


def _extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from model output.
    Returns a cleaned numeric string or None if no number found.
    Handles: integers, decimals, negatives, comma-formatted numbers.
    """
    # Prefer "The answer is X" pattern
    match = re.search(r'[Tt]he answer is:?\s*([\-]?\d+[\.,]?\d*)', text)
    if match:
        return match.group(1).replace(',', '')

    # Fallback: last number in text
    numbers = _NUMBER_PATTERN.findall(text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def _answers_match(pred_str: Optional[str], gold_str: str) -> bool:
    """
    Numeric comparison of predicted and gold answers.
    Handles integer vs float representations.
    """
    if pred_str is None:
        return False
    try:
        return abs(float(pred_str) - float(gold_str)) < 1e-4
    except ValueError:
        return pred_str.strip() == gold_str.strip()


# ── Training dataset ──────────────────────────────────────────────────────────

class GSM8KTrainDataset(Dataset):
    """
    Fine-tuning dataset for GSM8K / MetaMathQA.

    Format:
        "Problem: {question}\nSolution: {solution}\n"

    The model is trained to generate the full solution including the final
    "The answer is X." sentence.  This teaches chain-of-thought reasoning.

    MetaMathQA format:
        {"query": "...", "response": "...", "type": "..."}
    GSM8K format:
        {"question": "...", "answer": "...(####number)"}
    """

    def __init__(
        self,
        raw_dataset,
        tokenizer:  PreTrainedTokenizer,
        max_length: int = 512,
        dataset_type: str = "gsm8k",
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        for ex in raw_dataset:
            text = self._format(ex, dataset_type)
            if text:
                self.samples.append(text)

        logger.info(f"GSM8KTrainDataset: {len(self.samples)} training examples")

    def _format(self, ex: dict, dataset_type: str) -> Optional[str]:
        try:
            if dataset_type == "metamath":
                q = ex['query']
                a = ex['response']
            else:  # gsm8k
                q = ex['question']
                # GSM8K answers contain #### as separator
                a_parts = ex['answer'].split('####')
                steps   = a_parts[0].strip()
                final   = a_parts[1].strip() if len(a_parts) > 1 else ""
                a = f"{steps}\nThe answer is {final}." if final else steps

            return f"Problem: {q}\nSolution: {a}\n"
        except (KeyError, IndexError):
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        enc  = self.tokenizer(
            text,
            truncation   = True,
            max_length   = self.max_length,
            padding      = 'max_length',
            return_tensors = 'pt',
        )
        input_ids      = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Main data module ──────────────────────────────────────────────────────────

class GSM8KDataModule:
    """
    Data module for GSM8K mathematical reasoning.

    Args:
        tokenizer:      LLaMA tokenizer.
        use_metamath:   If True, use MetaMathQA for training (recommended).
                        MetaMathQA has 395K examples (only use a subset by default).
        metamath_subset: Number of MetaMathQA examples to use (None = all).
        max_length:     Truncation length for training sequences.
        batch_size:     Micro-batch size for training.
    """

    def __init__(
        self,
        tokenizer:       PreTrainedTokenizer,
        use_metamath:    bool = True,
        metamath_subset: Optional[int] = 50_000,
        max_length:      int  = 512,
        batch_size:      int  = 4,
        num_workers:     int  = 2,
    ) -> None:
        self.tokenizer   = tokenizer
        self.batch_size  = batch_size
        self.num_workers = num_workers

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

        if use_metamath:
            logger.info("Loading MetaMathQA for GSM8K training ...")
            raw_train = load_dataset("meta-math/MetaMathQA", split="train")
            if metamath_subset:
                raw_train = raw_train.select(range(min(metamath_subset, len(raw_train))))
            self._train_dataset = GSM8KTrainDataset(
                raw_train, tokenizer, max_length, dataset_type="metamath"
            )
        else:
            logger.info("Loading GSM8K train split ...")
            raw_train = load_dataset("gsm8k", "main", split="train")
            self._train_dataset = GSM8KTrainDataset(
                raw_train, tokenizer, max_length, dataset_type="gsm8k"
            )

        # GSM8K test set for evaluation (1319 examples)
        logger.info("Loading GSM8K test split ...")
        raw_test = load_dataset("gsm8k", "main", split="test")
        self._test_examples = [
            {"question": ex["question"], "answer": ex["answer"].split("####")[-1].strip()}
            for ex in raw_test
        ]
        logger.info(
            f"GSM8K: train={len(self._train_dataset)}  test={len(self._test_examples)}"
        )

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    @torch.no_grad()
    def evaluate_generation(
        self,
        model,
        device:         torch.device,
        max_new_tokens: int = 256,
        n_examples:     Optional[int] = None,
    ) -> float:
        """
        Evaluate accuracy by greedy generation on GSM8K test set.

        For each test question:
          1. Generate solution using greedy decoding.
          2. Extract final numeric answer.
          3. Compare to gold answer (numeric exact match).

        Returns accuracy in [0, 1].

        Args:
            model:          LLaMA model with PAFT v_proj (or any baseline).
            device:         Device to run on.
            max_new_tokens: Max tokens to generate per example.
            n_examples:     If set, evaluate on only the first n examples
                            (useful for quick sanity checks during training).
        """
        model.eval()
        correct, total = 0, 0

        examples = self._test_examples[:n_examples] if n_examples else self._test_examples

        for ex in examples:
            prompt = f"Problem: {ex['question']}\nSolution:"
            enc    = self.tokenizer(
                prompt,
                return_tensors    = 'pt',
                truncation        = True,
                max_length        = 512,
                add_special_tokens = True,
            )
            input_ids      = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)

            generated = model.generate(
                input_ids        = input_ids,
                attention_mask   = attention_mask,
                max_new_tokens   = max_new_tokens,
                do_sample        = False,
                temperature      = 1.0,
                pad_token_id     = self.tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            gen_text = self.tokenizer.decode(
                generated[0][input_ids.shape[1]:],
                skip_special_tokens = True,
            )
            pred = _extract_answer(gen_text)
            if _answers_match(pred, ex['answer']):
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def num_train_samples(self) -> int:
        return len(self._train_dataset)

    def num_test_samples(self) -> int:
        return len(self._test_examples)
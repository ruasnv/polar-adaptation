"""
Commonsense reasoning data module for LLaMA experiments.

The exact 8-dataset suite used in PoLAR (and standard in the PEFT literature):
  BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, OBQA

Training:   Next-token prediction (causal LM cross-entropy) on the training split.
            Each example is formatted as a multiple-choice question + answer.
Evaluation: Log-likelihood scoring of each choice — the choice with the highest
            log-likelihood of the answer tokens is the prediction.
            This matches the lm-evaluation-harness protocol used in PoLAR.

Format for training (causal LM):
    "Question: {question}\nAnswer: {answer}\n"

Format for evaluation (log-likelihood):
    Compute log p(answer_i | question) for each choice i.
    Predict the choice with highest log p.
    This is equivalent to lm-evaluation-harness 'multiple_choice' task type.

HF dataset keys per task:
  boolq:       question, passage, answer (bool → "yes"/"no")
  piqa:        goal, sol1, sol2, label (0 or 1 → sol1 or sol2)
  siqa:        context, question, answerA, answerB, answerC, label (1/2/3)
  hellaswag:   ctx_a, ctx_b, endings (list[4]), label
  winogrande:  sentence, option1, option2, answer ("1" or "2")
  arc_easy:    question, choices.text/label, answerKey
  arc_challenge: same as arc_easy
  openbookqa:  question_stem, choices.text/label, answerKey

Usage:
    dm = CommonsenseDataModule("boolq", tokenizer, max_length=256, batch_size=8)
    train_loader = dm.get_train_loader()
    val_loader   = dm.get_val_loader()   # for log-likelihood eval
    accuracy = dm.evaluate_log_likelihood(model, device)
"""

from __future__ import annotations

import logging
import string
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# ── Task configuration ────────────────────────────────────────────────────────

SUPPORTED_TASKS = [
    "boolq", "hellaswag",
    "winogrande", "arc_easy", "arc_challenge", "openbookqa"
]

# Maps task → HF dataset name/config
HF_NAMES: Dict[str, Tuple[str, Optional[str]]] = {
    "boolq":         ("google/boolq",   None),
    "piqa":          ("piqa",            None),
    "siqa":          ("social_i_qa",     None),
    "hellaswag":     ("Rowan/hellaswag",     None),
    "winogrande":    ("allenai/winogrande",  "winogrande_xl"),
    "arc_easy":      ("allenai/ai2_arc",     "ARC-Easy"),
    "arc_challenge": ("allenai/ai2_arc",     "ARC-Challenge"),
    "openbookqa":    ("allenai/openbookqa",  "main"),
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_boolq(ex) -> Tuple[str, List[str], int]:
    question = f"Passage: {ex['passage']}\nQuestion: {ex['question']}?"
    choices  = ["no", "yes"]
    label    = int(ex['answer'])
    return question, choices, label


def _format_piqa(ex) -> Tuple[str, List[str], int]:
    question = f"Goal: {ex['goal']}"
    choices  = [ex['sol1'], ex['sol2']]
    label    = int(ex['label'])
    return question, choices, label


def _format_siqa(ex) -> Tuple[str, List[str], int]:
    question = f"Context: {ex['context']}\nQuestion: {ex['question']}"
    choices  = [ex['answerA'], ex['answerB'], ex['answerC']]
    label    = int(ex['label']) - 1   # siqa labels are 1-indexed
    return question, choices, label


def _format_hellaswag(ex) -> Tuple[str, List[str], int]:
    ctx = ex['ctx_a'] + " " + ex['ctx_b']
    question = f"{ctx}"
    choices  = list(ex['endings'])
    label    = int(ex['label'])
    return question, choices, label


def _format_winogrande(ex) -> Tuple[str, List[str], int]:
    question = ex['sentence']
    choices  = [ex['option1'], ex['option2']]
    label    = int(ex['answer']) - 1   # winogrande labels: "1" or "2"
    return question, choices, label


def _format_arc(ex) -> Tuple[str, List[str], int]:
    question = ex['question']
    choices  = ex['choices']['text']
    labels   = ex['choices']['label']   # ["A","B","C","D"] or ["1","2","3","4"]
    answer   = ex['answerKey']
    # Map answer key to index
    try:
        if answer in labels:
            label = labels.index(answer)
        elif answer in string.ascii_uppercase:
            label = ord(answer) - ord('A')
        else:
            label = int(answer) - 1
    except (ValueError, IndexError):
        label = 0
    return question, choices, label


def _format_openbookqa(ex) -> Tuple[str, List[str], int]:
    question = ex['question_stem']
    choices  = ex['choices']['text']
    labels   = ex['choices']['label']   # ["A","B","C","D"]
    answer   = ex['answerKey']
    label    = labels.index(answer) if answer in labels else ord(answer) - ord('A')
    return question, choices, label


FORMATTERS = {
    "boolq":         _format_boolq,
    "piqa":          _format_piqa,
    "siqa":          _format_siqa,
    "hellaswag":     _format_hellaswag,
    "winogrande":    _format_winogrande,
    "arc_easy":      _format_arc,
    "arc_challenge": _format_arc,
    "openbookqa":    _format_openbookqa,
}


# ── Training dataset  (next-token prediction on answer) ───────────────────────

class CommonsenseLMDataset(Dataset):
    """
    Causal LM dataset: input = "Question: {q}\nAnswer: {a}\n"
    Labels = shifted input_ids (standard CLM convention — HF handles this
    automatically when labels == input_ids with -100 masking on pad tokens).

    The model is trained to predict the FULL sequence including the question,
    but the primary learning signal comes from the answer tokens.
    This matches the fine-tuning protocol in PoLAR and LLaMA-Adapter papers.
    """

    def __init__(
        self,
        raw_dataset,
        formatter,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        for ex in raw_dataset:
            try:
                question, choices, label = formatter(ex)
                answer = choices[label]
                text   = f"Question: {question}\nAnswer: {answer}\n"
                self.samples.append(text)
            except (KeyError, IndexError, TypeError):
                continue   # skip malformed examples

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
        # For CLM: labels = input_ids with -100 on padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Evaluation dataset  (log-likelihood multiple choice) ─────────────────────

class CommonsenseEvalDataset(Dataset):
    """
    Stores (question, choices, label) tuples for log-likelihood evaluation.
    Not tokenised here — tokenised batch-by-batch in evaluate_log_likelihood().
    """

    def __init__(self, raw_dataset, formatter) -> None:
        self.samples = []
        for ex in raw_dataset:
            try:
                q, choices, label = formatter(ex)
                self.samples.append((q, choices, label))
            except (KeyError, IndexError, TypeError):
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ── Main data module ──────────────────────────────────────────────────────────

class CommonsenseDataModule:
    """
    Data module for one commonsense reasoning dataset.

    Args:
        task_name:   One of SUPPORTED_TASKS.
        tokenizer:   LLaMA tokenizer.
        max_length:  Token truncation length for training (256 is sufficient).
        batch_size:  Micro-batch size for training.
        eval_batch_size: Batch size for log-likelihood evaluation.
    """

    def __init__(
        self,
        task_name:      str,
        tokenizer:      PreTrainedTokenizer,
        max_length:     int = 256,
        batch_size:     int = 8,
        eval_batch_size: int = 16,
        num_workers:    int = 2,
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
        self.eval_batch_size = eval_batch_size
        self.num_workers    = num_workers

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

        hf_name, hf_config = HF_NAMES[task_name]
        formatter = FORMATTERS[task_name]

        logger.info(f"Loading {task_name} ({hf_name}) ...")
        if hf_config:
            raw = load_dataset(hf_name, hf_config)
        else:
            raw = load_dataset(hf_name)

        self._train_lm   = CommonsenseLMDataset(
            raw["train"], formatter, tokenizer, max_length
        )
        val_split = "validation" if "validation" in raw else "test"
        self._val_eval   = CommonsenseEvalDataset(raw[val_split], formatter)

        logger.info(
            f"{task_name}: train={len(self._train_lm)} samples  "
            f"val={len(self._val_eval)} examples"
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_lm,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    # ── Log-likelihood evaluation ─────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_log_likelihood(self, model, device: torch.device) -> float:
        """
        Compute accuracy using log-likelihood scoring.

        For each example (question, choices, label):
          Compute log p(choice_i | question) for each choice i.
          Predict argmax_i log p(choice_i | question).
          Correct if prediction == label.

        This is the exact protocol used in PoLAR and lm-evaluation-harness.
        The model must be in eval mode and on the correct device.

        Returns:
            Accuracy as a float in [0, 1].
        """
        model.eval()
        correct, total = 0, 0

        for question, choices, label in self._val_eval:
            log_likelihoods = []

            for choice in choices:
                prompt = f"Question: {question}\nAnswer: "
                full   = prompt + choice + "\n"

                prompt_enc = self.tokenizer(
                    prompt, return_tensors='pt', add_special_tokens=True
                )
                full_enc = self.tokenizer(
                    full, return_tensors='pt', add_special_tokens=True
                )

                input_ids      = full_enc['input_ids'].to(device)
                attention_mask = full_enc['attention_mask'].to(device)
                prompt_len     = prompt_enc['input_ids'].shape[1]

                # Build labels: -100 for prompt tokens, answer tokens for rest
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100

                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels,
                )
                # outputs.loss is mean NLL over answer tokens — negate for log-likelihood
                log_likelihoods.append(-outputs.loss.item())

            pred = int(torch.tensor(log_likelihoods).argmax().item())
            correct += int(pred == label)
            total   += 1

        return correct / total if total > 0 else 0.0
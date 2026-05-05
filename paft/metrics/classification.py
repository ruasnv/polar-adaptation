"""
Classification metrics: accuracy and macro F1.

Used by post-training eval scripts for SST-2, ContractNLI, PubMedQA.

These are NOT called by the trainer (which only computes LM loss).
They are called by run_analysis.py and analysis/geometric_health.py
after training is complete, operating on the final model checkpoint.
"""

from __future__ import annotations

from typing import List

import torch
from sklearn.metrics import accuracy_score, f1_score


def accuracy(preds: List[int], labels: List[int]) -> float:
    """Accuracy for single-label classification."""
    return float(accuracy_score(labels, preds))


def macro_f1(preds: List[int], labels: List[int]) -> float:
    """Macro-averaged F1 across all classes."""
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def classification_metrics(preds: List[int], labels: List[int]) -> dict:
    """Return both accuracy and macro F1 together."""
    return {
        "accuracy": accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels),
    }


def predict_label_token(
    model,
    tokenizer,
    texts:       List[str],
    label_words: List[str],
    device:      torch.device,
    batch_size:  int = 16,
) -> List[int]:
    """
    Predict class labels by comparing the log-probability of each label word
    at the final token position of the prompt.

    This is the "channel" evaluation method for causal LMs:
    format text as "...\nLabel: " and score each candidate completion.

    Args:
        model:       GPT-2 model (or PAFT wrapper) in eval mode.
        tokenizer:   Matching tokeniser.
        texts:       List of prompt strings (without the label completion).
        label_words: List of candidate completions, e.g. ["positive","negative"].
        device:      Target device.
        batch_size:  Inference batch size.

    Returns:
        List of integer class indices (0 to len(label_words)-1).
    """
    model.eval()
    predictions = []

    # Pre-tokenise candidate label tokens (first token of each word)
    label_ids = [
        tokenizer.encode(" " + w, add_special_tokens=False)[0]
        for w in label_words
    ]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors   = "pt",
            padding          = True,
            truncation       = True,
            max_length       = 512,
        ).to(device)

        with torch.no_grad():
            out    = model(**enc)
            # Logits at the last non-padding token position for each sample
            logits = out.logits                          # [B, T, V]
            # Use the last real token (before padding)
            seq_lens = enc["attention_mask"].sum(dim=1) - 1  # [B]
            last_logits = logits[
                torch.arange(len(batch_texts)), seq_lens
            ]                                            # [B, V]

        label_logits = last_logits[:, label_ids]         # [B, n_labels]
        preds        = label_logits.argmax(dim=-1).cpu().tolist()
        predictions.extend(preds)

    return predictions
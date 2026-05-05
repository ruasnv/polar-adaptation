"""
Generation metrics: ROUGE and BLEU.

Used for summarisation tasks (CNN/DM, PubMed, BioASQ) and code generation.

These are post-hoc metrics — NOT computed during training.
Called by analysis scripts after checkpoints are loaded.
"""

from __future__ import annotations

from typing import List, Dict

import torch


def rouge_scores(
    predictions: List[str],
    references:  List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L.

    Args:
        predictions: List of generated / predicted strings.
        references:  List of reference strings (ground truth).

    Returns:
        {"rouge1": float, "rouge2": float, "rougeL": float}
        All values are F1 scores in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE metrics. "
            "Install with: pip install rouge-score"
        )

    scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals  = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n       = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    return {k: v / max(n, 1) for k, v in totals.items()}


def bleu_score(
    predictions: List[str],
    references:  List[str],
) -> float:
    """
    Corpus-level BLEU score.

    Args:
        predictions: List of hypothesis strings.
        references:  List of reference strings.

    Returns:
        BLEU score in [0, 1].
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        raise ImportError(
            "nltk is required for BLEU. Install with: pip install nltk"
        )

    smoother   = SmoothingFunction().method1
    hypothesis = [pred.split() for pred in predictions]
    reference  = [[ref.split()] for ref in references]
    return corpus_bleu(reference, hypothesis, smoothing_function=smoother)


def generate_completions(
    model,
    tokenizer,
    prompts:    List[str],
    max_new:    int         = 64,
    device:     torch.device = torch.device("cpu"),
    batch_size: int         = 8,
) -> List[str]:
    """
    Generate text completions for a list of prompt strings.

    Used to produce predictions for ROUGE/BLEU evaluation.

    Args:
        model:      GPT-2 model (or PAFT/SVF wrapper) in eval mode.
        tokenizer:  Matching tokeniser.
        prompts:    Input prompt strings.
        max_new:    Number of new tokens to generate beyond the prompt.
        device:     Target device.
        batch_size: Inference batch size.

    Returns:
        List of generated completion strings (prompt is NOT included).
    """
    model.eval()
    completions = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 448,  # leave room for max_new tokens within 512
        ).to(device)

        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens   = max_new,
                do_sample        = False,    # greedy for determinism
                pad_token_id     = tokenizer.pad_token_id,
                eos_token_id     = tokenizer.eos_token_id,
            )

        # Decode only the new tokens (after the prompt)
        new_ids  = out_ids[:, input_len:]
        decoded  = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        completions.extend(decoded)

    return completions
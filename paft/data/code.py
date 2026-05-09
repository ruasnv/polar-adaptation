"""
Code domain data module (extreme domain shift).

Tasks
─────
1. Code generation  — CodeSearchNet Python
   Dataset:  code_search_net  (config: python)
   Format:   "Docstring: {docstring}\nCode: {function_body}"
   Metric:   BLEU-4, Pass@1 (post-hoc on HumanEval)

Code is structurally unlike natural language — strict syntax, deterministic
semantics, deeply nested logical structure.  This is the maximum stress test
for geometry-preserving adaptation: if PAFT maintains geometric structure
even under the extreme code distribution, the non-additive claim is strong.

Dataset sizes:
    CodeSearchNet Python train: 412K    (capped at _MAX_TRAIN)
    CodeSearchNet Python valid: 13.8K   (capped at _MAX_VAL)
"""

from __future__ import annotations

import logging
from typing import List

from datasets import load_dataset

from paft.data.base import BaseDataModule

logger = logging.getLogger(__name__)

_MAX_TRAIN = 20_000
_MAX_VAL   =  2_000


class CodeDataModule(BaseDataModule):

    def _load_train_texts(self) -> List[str]:
        return _load_codesearchnet(split="train", max_samples=_MAX_TRAIN)

    def _load_val_texts(self) -> List[str]:
        return _load_codesearchnet(split="validation", max_samples=_MAX_VAL)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_codesearchnet(split: str, max_samples: int) -> List[str]:
    """
    Load CodeSearchNet Python and format as docstring → function.

    Format: "Docstring: {docstring}\nCode: {code}"

    Truncation strategy:
        - Docstring: first 80 words   (clear task description)
        - Code: first 300 words       (most of the function body)
    This leaves ~130 tokens for GPT-2's overhead, within the 512-token window.

    We use func_documentation_string (natural language) rather than
    docstring_tokens (pre-tokenised) because the natural text format
    matches how GPT-2 was pretrained.
    """
    logger.info(f"Loading CodeSearchNet Python ({split}, max={max_samples}) ...")
    ds = load_dataset("code_search_net", "python", split=split)
    texts = []
    for item in ds.select(range(min(max_samples, len(ds)))):
        docstring = item.get("func_documentation_string", "").strip()
        code      = item.get("func_code_string", "").strip()

        # Skip samples with empty docstring — no signal for the model
        if not docstring or not code:
            continue

        # Truncate both components
        docstring = " ".join(docstring.split()[:80])
        code      = " ".join(code.split()[:300])

        texts.append(f"Docstring: {docstring}\nCode: {code}")

    logger.info(f"  Loaded {len(texts)} code samples from {split}")
    return texts
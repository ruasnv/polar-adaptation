"""
domain_metrics.py — Axis 1: domain task performance evaluation.

Loads each trained model from checkpoint, runs it on the held-out test set
for its domain, and saves task-specific metrics to final/domain_metrics.json.

Metrics by domain
──────────────────
news       ROUGE-1/2/L on CNN/DailyMail test (summarisation)
legal      Accuracy + Macro-F1 on ContractNLI test (NLI)
biomedical Accuracy on PubMedQA labeled test (yes/no/maybe QA)
code       BLEU-4 on CodeSearchNet Python test (code generation)

These are NOT computed during training (training only saves eval_loss).
This script produces the main paper results table.

Usage
─────
    python analysis/domain_metrics.py --model gpt2_small --domain news
    python analysis/domain_metrics.py --model gpt2_small  # all domains
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis._utils import (
    load_trained_model, get_hf_model, get_tokenizer,
    discover_complete_runs, load_init_config,
)
from paft.metrics.generation    import rouge_scores, bleu_score, generate_completions
from paft.metrics.classification import classification_metrics, predict_label_token

ALL_METHODS = [
    "frozen","bitfit","svf","pure_paft","safe_pure_paft",
    "lora_r8","polar","hybrid_paft","safe_hybrid_paft","lora_r64","full_finetune",
]

# ──────────────────────────────────────────────────────────────────────────────
# Test data loaders (separate from training data)
# ──────────────────────────────────────────────────────────────────────────────

def load_news_test(max_samples: int = 250):
    from datasets import load_dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    prompts, references = [], []
    for item in ds.select(range(min(max_samples, len(ds)))):
        article  = " ".join(item["article"].strip().replace("\n"," ").split()[:300])
        summary  = item["highlights"].strip().replace("\n"," ")
        prompts.append(f"Article: {article}\nSummary:")
        references.append(summary)
    return prompts, references

def load_legal_test(max_samples: int = 500):
    from datasets import load_dataset
    _LABEL_MAP = {0:"entailment",1:"neutral",2:"contradiction",
                  "entailment":"entailment","neutral":"neutral",
                  "contradiction":"contradiction","not_mentioned":"neutral"}
    try:
        ds = load_dataset("kiddouk/contract-nli", split="test")
    except Exception:
        ds = load_dataset("contract_nli", split="test")
    prompts, labels, label_words = [], [], ["entailment","neutral","contradiction"]
    for item in ds.select(range(min(max_samples, len(ds)))):
        premise    = " ".join(item.get("premise","").strip().split()[:350])
        hypothesis = item.get("hypothesis","").strip()
        raw_label  = item.get("label", 1)
        label_text = _LABEL_MAP.get(raw_label, "neutral")
        prompts.append(f"Contract: {premise}\nClaim: {hypothesis}\nVerdict:")
        labels.append(label_words.index(label_text))
    return prompts, labels, label_words

def load_biomedical_test(max_samples: int = 500):
    from datasets import load_dataset
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    prompts, labels, label_words = [], [], ["yes","no","maybe"]
    for item in ds.select(range(min(max_samples, len(ds)))):
        background = " ".join(item.get("long_answer","").strip().split()[:250])
        question   = item["question"].strip()
        decision   = item["final_decision"].strip()
        prompts.append(f"Background: {background}\nQuestion: {question}\nAnswer:")
        if decision in label_words:
            labels.append(label_words.index(decision))
        else:
            labels.append(1)   # default "no" for unknown
    return prompts, labels, label_words

def load_code_test(max_samples: int = 250):
    from datasets import load_dataset
    ds = load_dataset("code_search_net", "python", split="test")
    prompts, references = [], []
    for item in ds.select(range(min(max_samples, len(ds)))):
        docstring = " ".join(item.get("func_documentation_string","").strip().split()[:80])
        code      = item.get("func_code_string","").strip()
        if not docstring or not code:
            continue
        prompts.append(f"Docstring: {docstring}\nCode:")
        references.append(" ".join(code.split()[:200]))
    return prompts, references

# ──────────────────────────────────────────────────────────────────────────────
# Per-domain evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_news(model, tokenizer, device) -> Dict:
    print("    Loading CNN/DM test set ...")
    prompts, references = load_news_test(max_samples=250)
    print(f"    Generating {len(prompts)} summaries ...")
    preds = generate_completions(model, tokenizer, prompts,
                                 max_new=64, device=device, batch_size=4)
    scores = rouge_scores(preds, references)
    print(f"    ROUGE-1={scores['rouge1']:.3f}  ROUGE-2={scores['rouge2']:.3f}  "
          f"ROUGE-L={scores['rougeL']:.3f}")
    return scores

def evaluate_legal(model, tokenizer, device) -> Dict:
    print("    Loading ContractNLI test set ...")
    prompts, labels, label_words = load_legal_test(max_samples=500)
    print(f"    Classifying {len(prompts)} contract clauses ...")
    preds = predict_label_token(model, tokenizer, prompts,
                                label_words, device, batch_size=16)
    scores = classification_metrics(preds, labels)
    print(f"    Accuracy={scores['accuracy']:.3f}  Macro-F1={scores['macro_f1']:.3f}")
    return scores

def evaluate_biomedical(model, tokenizer, device) -> Dict:
    print("    Loading PubMedQA test set ...")
    prompts, labels, label_words = load_biomedical_test(max_samples=500)
    print(f"    Classifying {len(prompts)} QA pairs ...")
    preds = predict_label_token(model, tokenizer, prompts,
                                label_words, device, batch_size=16)
    scores = classification_metrics(preds, labels)
    print(f"    Accuracy={scores['accuracy']:.3f}  Macro-F1={scores['macro_f1']:.3f}")
    return scores

def evaluate_code(model, tokenizer, device) -> Dict:
    print("    Loading CodeSearchNet test set ...")
    prompts, references = load_code_test(max_samples=250)
    print(f"    Generating {len(prompts)} code completions ...")
    preds = generate_completions(model, tokenizer, prompts,
                                 max_new=128, device=device, batch_size=4)
    bleu = bleu_score(preds, references)
    print(f"    BLEU-4={bleu:.3f}")
    return {"bleu4": round(bleu, 4)}

DOMAIN_EVALUATORS = {
    "news":       evaluate_news,
    "legal":      evaluate_legal,
    "biomedical": evaluate_biomedical,
    "code":       evaluate_code,
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(model: str, domains: List[str], checkpoint_root: Path, device: torch.device):
    runs = discover_complete_runs(checkpoint_root, model, domains, ALL_METHODS)
    print(f"\n=== Domain Metrics (Axis 1): {model} — {len(runs)} runs ===")

    for domain, method, run_dir in runs:
        out_path = run_dir / "final" / "domain_metrics.json"
        if out_path.exists():
            print(f"  SKIP (exists): {domain}/{method}")
            continue

        print(f"\n  Evaluating: {domain}/{method}")
        evaluator = DOMAIN_EVALUATORS.get(domain)
        if evaluator is None:
            print(f"    No evaluator for domain '{domain}' — skipping")
            continue

        try:
            cfg       = load_init_config(run_dir)
            method_obj= load_trained_model(run_dir, cfg, device)
            hf_model  = get_hf_model(method_obj)
            tokenizer = get_tokenizer(cfg["model"]["hf_name"])
            hf_model.eval()

            scores = evaluator(hf_model, tokenizer, device)
            scores["domain"] = domain
            scores["method"] = method

            with out_path.open("w") as f:
                json.dump(scores, f, indent=2)
            print(f"    Saved → {out_path}")

            method_obj.cleanup()

        except Exception as e:
            print(f"    ERROR: {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2_small")
    p.add_argument("--domains", nargs="+", default=["news","legal","biomedical","code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.domains,
        Path(args.checkpoint_root), torch.device(args.device))
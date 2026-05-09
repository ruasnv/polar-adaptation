"""
base_metrics.py — Axis 2: base model capability preservation (forgetting score).

For every trained model, evaluates on benchmarks it was NEVER trained on.
The forgetting score is the paper's core Axis 2 number.

Benchmarks (via lm-evaluation-harness)
───────────────────────────────────────
hellaswag        Commonsense completion    acc_norm (0-shot)
arc_easy         Elementary reasoning      acc_norm (0-shot)
lambada_openai   Long-range dependency     acc      (0-shot)
wikitext         Language modelling        word_perplexity

Forgetting score
────────────────
For perplexity metrics (lower=better):  retention = pretrained_ppl / adapted_ppl
For accuracy metrics  (higher=better):  retention = adapted_acc / pretrained_acc
forgetting_score = geometric_mean(all retentions)
1.0 = no forgetting, < 1.0 = degraded, > 1.0 = improved

Usage
─────
    pip install lm-eval>=0.4.0
    python analysis/base_metrics.py --model gpt2_small
    python analysis/base_metrics.py --model gpt2_small --pretrained_only
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis._utils import (
    load_trained_model, get_hf_model, get_tokenizer,
    discover_complete_runs, load_init_config,
)

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    _LM_EVAL = True
except ImportError:
    _LM_EVAL = False
    print("WARNING: lm-eval not installed.  Install with: pip install lm-eval>=0.4.0")

ALL_METHODS = [
    "frozen","bitfit","svf","pure_paft","safe_pure_paft",
    "lora_r8","polar","hybrid_paft","safe_hybrid_paft","lora_r64","full_finetune",
]

LM_EVAL_TASKS = ["hellaswag", "arc_easy", "lambada_openai", "wikitext"]

# ──────────────────────────────────────────────────────────────────────────────
# lm-eval evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_lm_eval(hf_model, tokenizer, device: torch.device) -> Dict[str, float]:
    """
    Run lm-evaluation-harness benchmarks on a loaded model.
    Returns flat dict of metric name → value.
    """
    if not _LM_EVAL:
        raise RuntimeError("lm-eval not installed. pip install lm-eval>=0.4.0")

    lm = HFLM(
        pretrained  = hf_model,
        tokenizer   = tokenizer,
        device      = str(device),
        batch_size  = 8,
    )
    results = lm_eval.simple_evaluate(
        model       = lm,
        tasks       = LM_EVAL_TASKS,
        num_fewshot = 0,
        log_samples = False,
    )

    # Extract the key metric per task
    flat = {}
    task_results = results.get("results", {})

    metric_map = {
        "hellaswag":      ("acc_norm,none", "acc_norm"),
        "arc_easy":       ("acc_norm,none", "acc_norm"),
        "lambada_openai": ("acc,none",      "acc"),
        "wikitext":       ("word_perplexity,none", "word_perplexity"),
    }
    for task, (key, short) in metric_map.items():
        if task in task_results:
            val = task_results[task].get(key, task_results[task].get(short))
            if val is not None:
                flat[task] = round(float(val), 4)

    return flat

# ──────────────────────────────────────────────────────────────────────────────
# Forgetting score
# ──────────────────────────────────────────────────────────────────────────────

_HIGHER_IS_BETTER = {
    "hellaswag":      True,
    "arc_easy":       True,
    "lambada_openai": True,
    "wikitext":       False,   # lower perplexity = better
}

def compute_forgetting_score(
    pretrained_metrics: Dict[str, float],
    adapted_metrics:    Dict[str, float],
) -> float:
    """
    Geometric mean of per-benchmark retention scores.
    1.0 = perfect retention, < 1.0 = forgetting.
    """
    retentions = []
    for task, higher_is_better in _HIGHER_IS_BETTER.items():
        pre  = pretrained_metrics.get(task)
        post = adapted_metrics.get(task)
        if pre is None or post is None or pre == 0:
            continue
        if higher_is_better:
            retention = post / pre          # acc: want >= 1.0
        else:
            retention = pre / post          # ppl: want >= 1.0 (pre/post since lower=better)
        retentions.append(max(retention, 1e-6))   # floor at near-zero

    if not retentions:
        return float("nan")
    log_mean = sum(math.log(r) for r in retentions) / len(retentions)
    return round(math.exp(log_mean), 4)

# ──────────────────────────────────────────────────────────────────────────────
# Pretrained baseline (run once)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_pretrained(
    hf_name:    str,
    device:     torch.device,
    cache_path: Path,
) -> Dict[str, float]:
    """
    Evaluate the stock pretrained GPT-2 on all benchmarks.
    Results are cached to avoid re-running for every adapted model.
    """
    if cache_path.exists():
        with cache_path.open() as f:
            print(f"  Loaded pretrained baseline from cache: {cache_path}")
            return json.load(f)

    print(f"  Evaluating pretrained {hf_name} (run once, cached) ...")
    from transformers import GPT2LMHeadModel
    model     = GPT2LMHeadModel.from_pretrained(hf_name).to(device).eval()
    tokenizer = get_tokenizer(hf_name)

    metrics = run_lm_eval(model, tokenizer, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Pretrained metrics: {metrics}")
    print(f"  Cached → {cache_path}")
    return metrics

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(
    model:            str,
    domains:          List[str],
    checkpoint_root:  Path,
    figure_dir:       Path,
    device:           torch.device,
    pretrained_only:  bool = False,
):
    if not _LM_EVAL:
        print("ERROR: lm-eval required.  pip install lm-eval>=0.4.0")
        return

    print(f"\n=== Base Metrics (Axis 2): {model} ===")

    # Load any complete run's config to get hf_name
    all_runs = discover_complete_runs(checkpoint_root, model, domains, ALL_METHODS)
    if not all_runs:
        print("No complete runs found.")
        return

    _, _, sample_run_dir = all_runs[0]
    sample_cfg = load_init_config(sample_run_dir)
    hf_name    = sample_cfg["model"]["hf_name"]

    # Evaluate pretrained baseline (cached)
    cache_path = figure_dir / f"pretrained_baseline_{model}.json"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pretrained_metrics = evaluate_pretrained(hf_name, device, cache_path)

    if pretrained_only:
        print("--pretrained_only: done.")
        return

    # Evaluate each trained model
    for domain, method, run_dir in all_runs:
        out_path = run_dir / "final" / "base_metrics.json"
        if out_path.exists():
            print(f"  SKIP (exists): {domain}/{method}")
            continue

        print(f"\n  Evaluating: {domain}/{method}")
        try:
            cfg        = load_init_config(run_dir)
            method_obj = load_trained_model(run_dir, cfg, device)
            hf_model   = get_hf_model(method_obj)
            tokenizer  = get_tokenizer(hf_name)
            hf_model.eval()

            adapted_metrics    = run_lm_eval(hf_model, tokenizer, device)
            forgetting_score   = compute_forgetting_score(
                pretrained_metrics, adapted_metrics
            )

            result = {
                "method":            method,
                "domain":            domain,
                "pretrained":        pretrained_metrics,
                "adapted":           adapted_metrics,
                "forgetting_score":  forgetting_score,
            }
            with out_path.open("w") as f:
                json.dump(result, f, indent=2)
            print(f"    Forgetting score: {forgetting_score:.4f}")
            print(f"    Saved → {out_path}")

            method_obj.cleanup()

        except Exception as e:
            print(f"    ERROR: {e}")

    # Aggregate forgetting scores into a summary table
    _write_forgetting_summary(checkpoint_root, model, domains, figure_dir)

def _write_forgetting_summary(
    checkpoint_root: Path,
    model: str, domains: List[str], figure_dir: Path
):
    import csv
    rows = []
    for domain in domains:
        for method in ALL_METHODS:
            path = checkpoint_root / model / domain / method / "final" / "base_metrics.json"
            if not path.exists():
                continue
            with path.open() as f:
                data = json.load(f)
            row = {"model": model, "domain": domain, "method": method,
                   "forgetting_score": data.get("forgetting_score", "")}
            row.update({f"adapted_{k}": v
                        for k, v in data.get("adapted", {}).items()})
            rows.append(row)
    if not rows:
        return
    csv_path = figure_dir / f"forgetting_summary_{model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved forgetting summary → {csv_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2_small")
    p.add_argument("--domains", nargs="+", default=["news","biomedical","code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/base_metrics")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrained_only", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.domains, Path(args.checkpoint_root),
        Path(args.figure_dir), torch.device(args.device), args.pretrained_only)
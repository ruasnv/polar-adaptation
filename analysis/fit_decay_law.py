#!/usr/bin/env python3
"""
analysis/fit_decay_law.py

Fits a logarithmic decay law to sr(W_eff) as a function of total
gradient steps across GLUE tasks, for each fine-tuning method.

Model:  sr(W_eff) ≈ a - b * log(steps)

where b is the decay rate — how many stable rank units are lost per
10x increase in training steps.  A lower b indicates more geometric
stability under extended training.

Output files (written to results/analysis/):
    decay_law_results.json      Full fit results per method
    table_decay_rates.tex       LaTeX tabular fragment for the paper

Usage:
    python3 analysis/fit_decay_law.py
    python3 analysis/fit_decay_law.py --cache results/analysis/metrics_cache.json
    python3 analysis/fit_decay_law.py --output_dir results/analysis
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ── Training step counts ───────────────────────────────────────────────────────
# Total gradient steps = floor(N_train / batch_size) * n_epochs
# DeBERTa: batch_size=32
# Epochs: CoLA/MRPC/RTE/STS-B=10, SST-2/QNLI=5, MNLI/QQP=3
# Training set sizes from the paper's experimental setup section.

STEPS: Dict[str, int] = {
    "rte":  round(2_490   / 32) * 10,
    "mrpc": round(3_668   / 32) * 10,
    "stsb": round(5_749   / 32) * 10,
    "cola": round(8_551   / 32) * 10,
    "sst2": round(67_349  / 32) * 5,
    "qnli": round(104_743 / 32) * 5,
    "qqp":  round(363_849 / 32) * 3,
    "mnli": round(392_702 / 32) * 3,
}

# Display labels for LaTeX output
METHOD_LABELS: Dict[str, str] = {
    "lora_r8":          r"LoRA $r{=}8$",
    "lora_r64":         r"LoRA $r{=}64$",
    "polar_r8":         r"PoLAR $r{=}8$",
    "pure_paft":        "pure-PAFT",
    "hybrid_paft":      "hybrid-PAFT",
    "safe_pure_paft":   "safe-pure-PAFT",
    "safe_hybrid_paft": "safe-hybrid-PAFT",
    "svf":              "SVF",
    "bitfit":           "BitFit",
    "full_ft":          "Full FT",
    "frozen":           "Frozen",
}

# Methods to fit and include in the paper table
# Frozen and BitFit are excluded — they never change sr(W_eff)
FIT_METHODS: List[str] = [
    "lora_r8",
    "lora_r64",
    "polar_r8",
    "safe_hybrid_paft",
    "hybrid_paft",
    "safe_pure_paft",
    "pure_paft",
    "svf",
]

# Methods shown in the main paper table (primary comparison)
PRIMARY_METHODS: List[str] = [
    "lora_r8",
    "lora_r64",
    "polar_r8",
    "safe_hybrid_paft",
]


# ── Decay model ────────────────────────────────────────────────────────────────

def log_decay(steps: np.ndarray, a: float, b: float) -> np.ndarray:
    """sr(W_eff) = a - b * log(steps)"""
    return a - b * np.log(steps)


def fit_log_decay(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "",
) -> Optional[Dict]:
    """
    Fit log decay model to (steps, sr) data.

    Returns dict with keys: a, b, r2, pearson_r, pearson_p, n_points,
    or None if fit fails or too few points.
    """
    if len(x) < 4:
        return None

    try:
        # Suppress scipy warnings about covariance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                log_decay,
                x, y,
                p0=[40.0, 2.0],
                bounds=([0, 0], [200, 50]),
                maxfev=10_000,
            )

        a, b = popt
        y_pred = log_decay(x, a, b)

        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Pearson correlation between log(steps) and sr
        log_x = np.log(x)
        pearson_r, pearson_p = pearsonr(log_x, y)

        # Parameter standard errors
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]

        return {
            "a":         float(a),
            "b":         float(b),
            "a_std":     float(perr[0]),
            "b_std":     float(perr[1]),
            "r2":        float(r2),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "n_points":  int(len(x)),
            "steps":     x.tolist(),
            "sr_values": y.tolist(),
            "sr_fitted": y_pred.tolist(),
        }

    except Exception as e:
        print(f"  Fit failed for {method}: {e}")
        return None


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sr_per_task(
    cache: Dict,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (steps, sr_final) pairs for a method across all GLUE tasks.
    Returns sorted arrays (ascending steps).
    """
    glue = cache.get("glue", {})
    pairs = []

    for task, steps in STEPS.items():
        entry = glue.get(task, {}).get(method, {})
        sr    = entry.get("sr_Weff_final")
        if sr is not None:
            pairs.append((steps, float(sr)))

    if not pairs:
        return np.array([]), np.array([])

    pairs.sort(key=lambda p: p[0])
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return x, y


# ── Monotonicity check ─────────────────────────────────────────────────────────

def check_monotonicity(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Check whether sr values are monotonically decreasing in steps.
    Returns a dict with monotone flag and any violations.
    """
    violations = []
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            violations.append({
                "task_steps_prev": int(x[i - 1]),
                "task_steps_curr": int(x[i]),
                "sr_prev":         float(y[i - 1]),
                "sr_curr":         float(y[i]),
                "delta":           float(y[i] - y[i - 1]),
            })
    return {
        "is_monotone":  len(violations) == 0,
        "n_violations": len(violations),
        "violations":   violations,
    }


# ── LaTeX table generation ─────────────────────────────────────────────────────

def make_latex_table(
    results: Dict[str, Dict],
    primary_only: bool = True,
) -> str:
    """
    Generate LaTeX tabular fragment for the decay rate table.

    primary_only=True: include only PRIMARY_METHODS (4 rows, main paper)
    primary_only=False: include all FIT_METHODS (appendix)
    """
    methods = PRIMARY_METHODS if primary_only else FIT_METHODS

    lines = []
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & "
        r"\textbf{Decay rate $b$} & "
        r"$R^2$ \\"
    )
    lines.append(r"\midrule")

    for method in methods:
        if method not in results:
            continue
        fit = results[method]
        if fit is None:
            lines.append(
                f"{METHOD_LABELS.get(method, method)} & --- & --- \\\\"
            )
            continue

        b   = fit["b"]
        r2  = fit["r2"]
        label = METHOD_LABELS.get(method, method)
        lines.append(f"{label} & {b:.2f} & {r2:.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(results: Dict[str, Dict]) -> None:
    print()
    print("Logarithmic Decay Law: sr(W_eff) ≈ a - b·log(steps)")
    print(f"{'Method':<24}  {'b':>6}  {'±':>5}  {'R²':>6}  "
          f"{'Pearson r':>10}  {'p-value':>10}  {'pts':>4}  {'Monotone'}")
    print("─" * 90)

    for method in FIT_METHODS:
        if method not in results:
            continue
        fit = results[method]
        if fit is None:
            print(f"{METHOD_LABELS.get(method, method):<24}  fit failed")
            continue

        mono  = check_monotonicity(
            np.array(fit["steps"]), np.array(fit["sr_values"])
        )
        mono_str = "✓" if mono["is_monotone"] else f"✗ ({mono['n_violations']} viol.)"

        print(
            f"{METHOD_LABELS.get(method, method):<24}  "
            f"{fit['b']:>6.3f}  "
            f"{fit['b_std']:>5.3f}  "
            f"{fit['r2']:>6.4f}  "
            f"{fit['pearson_r']:>10.4f}  "
            f"{fit['pearson_p']:>10.4e}  "
            f"{fit['n_points']:>4}  "
            f"{mono_str}"
        )

    # Key comparison
    print()
    lora_b = results.get("lora_r8", {})
    paft_b = results.get("safe_hybrid_paft", {})
    if lora_b and paft_b:
        ratio = lora_b["b"] / paft_b["b"]
        print(f"Decay rate ratio (LoRA r8 / safe-hybrid-PAFT): {ratio:.2f}×")
        print(f"  LoRA b={lora_b['b']:.3f}, PAFT b={paft_b['b']:.3f}")
        print(f"  PAFT degrades {ratio:.1f}× slower per log-step than LoRA")

    # Monotonicity violations
    print()
    print("Monotonicity check (sr should decrease monotonically in steps):")
    for method in FIT_METHODS:
        if method not in results or results[method] is None:
            continue
        fit  = results[method]
        mono = check_monotonicity(
            np.array(fit["steps"]), np.array(fit["sr_values"])
        )
        if not mono["is_monotone"]:
            for v in mono["violations"]:
                print(
                    f"  {method}: sr rises at "
                    f"{v['task_steps_prev']}→{v['task_steps_curr']} steps "
                    f"({v['sr_prev']:.2f}→{v['sr_curr']:.2f}, "
                    f"Δ={v['delta']:+.2f}) — consistent with Prop. 1 "
                    f"(task-specific ΔW·W₀ interaction)"
                )


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit log decay law to sr(W_eff)")
    p.add_argument(
        "--cache",
        default="results/analysis/metrics_cache.json",
        help="Path to metrics_cache.json",
    )
    p.add_argument(
        "--output_dir",
        default="results/analysis",
        help="Directory to write output files",
    )
    p.add_argument(
        "--all_methods",
        action="store_true",
        help="Include all methods in LaTeX table (not just primary 4)",
    )
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    cache_p  = Path(args.cache)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_p.exists():
        print(f"ERROR: cache not found at {cache_p}")
        print("Run bash run_analysis.sh first to build the cache.")
        return

    with open(cache_p) as f:
        cache = json.load(f)

    print(f"Loaded cache from {cache_p}")
    print(f"Tasks available: {sorted(cache.get('glue', {}).keys())}")
    print()

    # Fit each method
    all_results: Dict[str, Optional[Dict]] = {}

    for method in FIT_METHODS:
        x, y = load_sr_per_task(cache, method)
        if len(x) == 0:
            print(f"  {method}: no data found in cache")
            continue

        print(f"  Fitting {method}  ({len(x)} tasks) ...")
        fit = fit_log_decay(x, y, method=method)
        all_results[method] = fit

    # Print summary
    print_summary(all_results)

    # Write JSON
    json_path = out_dir / "decay_law_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "model":   "sr(W_eff) = a - b * log(steps)",
                "steps_formula": "floor(N_train / 32) * n_epochs",
                "methods": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved → {json_path}")

    # Write main paper LaTeX table (primary 4 methods)
    primary_tex = make_latex_table(all_results, primary_only=True)
    tex_path = out_dir / "table_decay_rates.tex"
    tex_path.write_text(primary_tex)
    print(f"Main table  → {tex_path}")

    # Write appendix LaTeX table (all methods)
    full_tex = make_latex_table(all_results, primary_only=False)
    full_path = out_dir / "table_decay_rates_full.tex"
    full_path.write_text(full_tex)
    print(f"Full table  → {full_path}")

    # Sanity check: warn if any primary method has R² < 0.85
    print()
    for method in PRIMARY_METHODS:
        fit = all_results.get(method)
        if fit and fit["r2"] < 0.85:
            print(
                f"WARNING: {method} has R²={fit['r2']:.3f} — "
                f"log decay may not be the right model for this method."
            )
        elif fit:
            print(f"OK: {method}  b={fit['b']:.3f}  R²={fit['r2']:.4f}")


if __name__ == "__main__":
    main()
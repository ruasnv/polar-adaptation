#!/usr/bin/env python3
"""
analysis/fit_decay_law_wO.py

W_O companion to fit_decay_law.py. Fits the same logarithmic decay law,
sr(W_O,eff) ~ a - b*log(steps), across GLUE tasks, for the projection this
paper reports separately from W_V.

Restricted to METHODS_WITH_REAL_O (same set as table_wO_metrics.py):
Frozen, BitFit, and LoRA are excluded here because their W_O is untouched
by construction — fitting a decay curve to a structural zero is meaningless,
not just uninformative.

Output files (written to results/analysis/):
    decay_law_results_wO.json      Full fit results per method
    table_decay_rates_wO.tex       LaTeX tabular fragment (primary methods)
    table_decay_rates_wO_full.tex  LaTeX tabular fragment (all methods)

Usage:
    python3 analysis/fit_decay_law_wO.py
    python3 analysis/fit_decay_law_wO.py --cache results/analysis/metrics_cache.json
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

# Same step counts as fit_decay_law.py — must match exactly, this is the
# same x-axis, just a different y-axis (W_O instead of W_V).
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

METHOD_LABELS: Dict[str, str] = {
    "polar_r8":         r"PoLAR $r{=}8$",
    "pure_paft":        "pure-PAFT",
    "hybrid_paft":      "hybrid-PAFT",
    "safe_pure_paft":   "safe-pure-PAFT",
    "safe_hybrid_paft": "safe-hybrid-PAFT",
    "svf":              "SVF",
    "full_ft":          "Full FT",
}

# Methods whose training mechanism actually reaches W_O — must match
# table_wO_metrics.py's METHODS_WITH_REAL_O exactly. Frozen, BitFit, and
# LoRA (DeBERTa target_modules = query/value only) are deliberately
# excluded, not just omitted by oversight.
FIT_METHODS: List[str] = [
    "svf",
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "polar_r8",
    "full_ft",
]

# LoRA is excluded from W_O entirely (untouched), so the primary-table
# comparison for W_O is necessarily a different subset than fit_decay_law.py's
# PRIMARY_METHODS. This set highlights the safe-vs-non-safe contrast plus
# the two non-PAFT methods that do touch W_O.
PRIMARY_METHODS: List[str] = [
    "safe_hybrid_paft",
    "pure_paft",
    "polar_r8",
    "svf",
]


def log_decay(steps: np.ndarray, a: float, b: float) -> np.ndarray:
    return a - b * np.log(steps)


def fit_log_decay(x: np.ndarray, y: np.ndarray, method: str = "") -> Optional[Dict]:
    if len(x) < 4:
        print(f"  {method}: fewer than 4 data points ({len(x)}) — fit skipped")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                log_decay, x, y,
                p0=[40.0, 2.0], bounds=([0, 0], [200, 50]), maxfev=10_000,
            )
        a, b = popt
        y_pred = log_decay(x, a, b)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        log_x = np.log(x)
        pearson_r, pearson_p = pearsonr(log_x, y)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
        return {
            "a": float(a), "b": float(b),
            "a_std": float(perr[0]), "b_std": float(perr[1]),
            "r2": float(r2), "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "n_points": int(len(x)), "steps": x.tolist(),
            "sr_values": y.tolist(), "sr_fitted": y_pred.tolist(),
        }
    except Exception as e:
        print(f"  Fit failed for {method}: {e}")
        return None


def load_sr_o_per_task(cache: Dict, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (steps, sr_O_final) pairs for a method across all GLUE tasks."""
    glue = cache.get("glue", {})
    pairs = []
    for task, steps in STEPS.items():
        entry = glue.get(task, {}).get(method, {})
        sr = entry.get("sr_Weff_O_final")
        if sr is not None:
            pairs.append((steps, float(sr)))
    if not pairs:
        return np.array([]), np.array([])
    pairs.sort(key=lambda p: p[0])
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def check_monotonicity(x: np.ndarray, y: np.ndarray) -> Dict:
    violations = []
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            violations.append({
                "task_steps_prev": int(x[i - 1]), "task_steps_curr": int(x[i]),
                "sr_prev": float(y[i - 1]), "sr_curr": float(y[i]),
                "delta": float(y[i] - y[i - 1]),
            })
    return {"is_monotone": len(violations) == 0, "n_violations": len(violations), "violations": violations}


def make_latex_table(results: Dict[str, Dict], primary_only: bool = True) -> str:
    methods = PRIMARY_METHODS if primary_only else FIT_METHODS
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"\textbf{Method} & \textbf{Decay rate $b_O$} & $R^2$ \\", r"\midrule"]
    for method in methods:
        if method not in results:
            continue
        fit = results[method]
        label = METHOD_LABELS.get(method, method)
        if fit is None:
            lines.append(f"{label} & --- & --- \\\\")
            continue
        lines.append(f"{label} & {fit['b']:.2f} & {fit['r2']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def print_summary(results: Dict[str, Dict]) -> None:
    print()
    print("Logarithmic Decay Law (W_O): sr(W_O,eff) ~ a - b*log(steps)")
    print("Restricted to methods whose training mechanism reaches W_O.")
    print(f"{'Method':<24}  {'b_O':>6}  {'±':>5}  {'R²':>6}  "
          f"{'Pearson r':>10}  {'p-value':>10}  {'pts':>4}  {'Monotone'}")
    print("-" * 90)
    for method in FIT_METHODS:
        if method not in results:
            continue
        fit = results[method]
        if fit is None:
            print(f"{METHOD_LABELS.get(method, method):<24}  fit failed / insufficient data")
            continue
        mono = check_monotonicity(np.array(fit["steps"]), np.array(fit["sr_values"]))
        mono_str = "check" if mono["is_monotone"] else f"x ({mono['n_violations']} viol.)"
        print(f"{METHOD_LABELS.get(method, method):<24}  "
              f"{fit['b']:>6.3f}  {fit['b_std']:>5.3f}  {fit['r2']:>6.4f}  "
              f"{fit['pearson_r']:>10.4f}  {fit['pearson_p']:>10.4e}  "
              f"{fit['n_points']:>4}  {mono_str}")

    print()
    safe = results.get("safe_hybrid_paft", {})
    pure = results.get("pure_paft", {})
    if safe and pure and safe.get("b") and pure.get("b"):
        ratio = pure["b"] / safe["b"]
        print(f"Decay rate ratio (pure-PAFT / safe-hybrid-PAFT, W_O): {ratio:.2f}x")

    print()
    print("Monotonicity check (sr should decrease monotonically in steps):")
    for method in FIT_METHODS:
        if method not in results or results[method] is None:
            continue
        fit = results[method]
        mono = check_monotonicity(np.array(fit["steps"]), np.array(fit["sr_values"]))
        if not mono["is_monotone"]:
            for v in mono["violations"]:
                print(f"  {method}: sr rises at {v['task_steps_prev']}->{v['task_steps_curr']} steps "
                      f"({v['sr_prev']:.2f}->{v['sr_curr']:.2f}, delta={v['delta']:+.2f})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit W_O log decay law")
    p.add_argument("--cache", default="results/analysis/metrics_cache.json")
    p.add_argument("--output_dir", default="results/analysis")
    p.add_argument("--all_methods", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_p = Path(args.cache)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_p.exists():
        print(f"ERROR: cache not found at {cache_p}")
        return

    with open(cache_p) as f:
        cache = json.load(f)

    print(f"Loaded cache from {cache_p}")

    all_results: Dict[str, Optional[Dict]] = {}
    for method in FIT_METHODS:
        x, y = load_sr_o_per_task(cache, method)
        if len(x) == 0:
            print(f"  {method}: no W_O data found in cache")
            continue
        print(f"  Fitting {method} (W_O, {len(x)} tasks) ...")
        all_results[method] = fit_log_decay(x, y, method=method)

    print_summary(all_results)

    json_path = out_dir / "decay_law_results_wO.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": "sr(W_O,eff) = a - b * log(steps)",
            "steps_formula": "floor(N_train / 32) * n_epochs",
            "scope": "W_O only, restricted to methods whose training mechanism reaches W_O",
            "methods": all_results,
        }, f, indent=2)
    print(f"\nResults saved -> {json_path}")

    primary_tex = make_latex_table(all_results, primary_only=True)
    tex_path = out_dir / "table_decay_rates_wO.tex"
    tex_path.write_text(primary_tex)
    print(f"Main table  -> {tex_path}")

    full_tex = make_latex_table(all_results, primary_only=False)
    full_path = out_dir / "table_decay_rates_wO_full.tex"
    full_path.write_text(full_tex)
    print(f"Full table  -> {full_path}")

    print()
    for method in FIT_METHODS:
        fit = all_results.get(method)
        if fit and fit["r2"] < 0.85:
            print(f"WARNING: {method} has R2={fit['r2']:.3f} — log decay may not fit W_O well for this method.")
        elif fit:
            print(f"OK: {method}  b_O={fit['b']:.3f}  R2={fit['r2']:.4f}")


if __name__ == "__main__":
    main()
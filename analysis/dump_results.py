#!/usr/bin/env python3
"""
dump_results.py

Reads metrics_cache.json and paft_cache.json and prints a structured
plain-text report you can copy-paste for review.

Usage:
    python3 dump_results.py
    python3 dump_results.py --cache results/analysis/metrics_cache.json \
                             --paft  results/analysis/paft_cache.json
"""
import json
import argparse
from pathlib import Path

TASK_PRIMARY = {
    "cola": "MCC", "mnli": "Acc", "mrpc": "F1",
    "qnli": "Acc", "qqp":  "F1", "rte":  "Acc",
    "sst2": "Acc", "stsb": "Pearson",
}
METHOD_ORDER = [
    "frozen", "bitfit", "svf",
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "lora_r8", "lora_r64", "polar_r8", "full_ft",
]
TASK_ORDER = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"]


def sep(char="─", width=90):
    print(char * width)


def section(title):
    sep("═")
    print(f"  {title}")
    sep("═")


def load(path, label):
    p = Path(path)
    if not p.exists():
        print(f"  [MISSING] {label}: {path}")
        return None
    with open(p) as f:
        return json.load(f)


def fmt(val, decimals=4):
    if val is None:
        return "  N/A  "
    return f"{float(val):.{decimals}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="results/analysis/metrics_cache.json")
    parser.add_argument("--paft",  default="results/analysis/paft_cache.json")
    args = parser.parse_args()

    cache = load(args.cache, "metrics_cache")
    paft  = load(args.paft,  "paft_cache")

    glue = cache.get("glue", {}) if cache else {}

    # ── 1. Performance table ───────────────────────────────────────────────
    section("TABLE 1: GLUE TASK PERFORMANCE")
    tasks_available = [t for t in TASK_ORDER if t in glue]
    header = f"{'Method':<22}" + "".join(f"  {t.upper():<8}" for t in tasks_available)
    print(header)
    sep()
    for method in METHOD_ORDER:
        row = f"{method:<22}"
        for task in tasks_available:
            score = glue.get(task, {}).get(method, {}).get("task_score")
            row += f"  {fmt(score)}"
        print(row)
    print()

    # ── 2. sr(W_eff) before/after ──────────────────────────────────────────
    section("TABLE 2: sr(W_eff) INIT vs FINAL — averaged across tasks")
    print(f"{'Method':<22}  {'sr_init':>8}  {'sr_final':>9}  {'delta':>8}  {'delta%':>8}")
    sep()
    for method in METHOD_ORDER:
        inits, finals = [], []
        for task in tasks_available:
            d = glue.get(task, {}).get(method, {})
            if d.get("sr_Weff_init") is not None:
                inits.append(d["sr_Weff_init"])
            if d.get("sr_Weff_final") is not None:
                finals.append(d["sr_Weff_final"])
        if not inits or not finals:
            print(f"{method:<22}  {'N/A':>8}  {'N/A':>9}  {'N/A':>8}  {'N/A':>8}")
            continue
        mean_i = sum(inits) / len(inits)
        mean_f = sum(finals) / len(finals)
        delta  = mean_f - mean_i
        pct    = delta / mean_i * 100 if mean_i else 0
        print(f"{method:<22}  {mean_i:>8.4f}  {mean_f:>9.4f}  {delta:>+8.4f}  {pct:>+7.2f}%")
    print()

    # ── 3. sr(W_eff) per task per method ──────────────────────────────────
    section("TABLE 3: sr(W_eff) FINAL — per task breakdown")
    header = f"{'Method':<22}" + "".join(f"  {t.upper():<8}" for t in tasks_available)
    print(header)
    sep()
    for method in METHOD_ORDER:
        row = f"{method:<22}"
        for task in tasks_available:
            v = glue.get(task, {}).get(method, {}).get("sr_Weff_final")
            row += f"  {fmt(v)}"
        print(row)
    print()

    # ── 4. sr(ΔW) per task per method ─────────────────────────────────────
    section("TABLE 4: sr(ΔW) — per task breakdown")
    header = f"{'Method':<22}" + "".join(f"  {t.upper():<8}" for t in tasks_available)
    print(header)
    sep()
    for method in METHOD_ORDER:
        row = f"{method:<22}"
        for task in tasks_available:
            v = glue.get(task, {}).get(method, {}).get("sr_deltaW_V")
            row += f"  {fmt(v)}"
        print(row)
    print()
    print("  Note: sr(ΔW) = N/A for methods that do not adapt weight matrices (frozen, bitfit).")
    print("        sr(ΔW) is undefined for the zero matrix.")

    # ── 5. Trainable parameter counts ─────────────────────────────────────
    section("TABLE 5: TRAINABLE PARAMETERS")
    print(f"{'Method':<22}  {'Params':>12}")
    sep()
    for method in METHOD_ORDER:
        # Use first available task
        params = None
        for task in tasks_available:
            p = glue.get(task, {}).get(method, {}).get("trainable_params")
            if p is not None and p > 0:
                params = p
                break
        val = f"{params:>12,}" if params else f"{'N/A':>12}"
        print(f"{method:<22}  {val}")
    print()

    # ── 6. Multi-metric summary ────────────────────────────────────────────
    section("TABLE 6: MULTI-METRIC SUMMARY — averaged across tasks")
    metrics = [
        ("sr_Weff_final",               "sr(W_eff)"),
        ("sr_deltaW_V",                 "sr(ΔW)"),
        ("spectral_entropy_Weff_final", "SpEnt"),
        ("effective_rank_Weff_final",   "EffRank"),
        ("condition_number_final",      "CondNum"),
        ("isotropy_final",              "Isotropy"),
        ("participation_ratio_final",   "PR"),
    ]
    header = f"{'Method':<22}" + "".join(f"  {label:>9}" for _, label in metrics)
    print(header)
    sep(width=100)
    for method in METHOD_ORDER:
        row = f"{method:<22}"
        for key, _ in metrics:
            vals = [glue[t][method][key]
                    for t in tasks_available
                    if method in glue.get(t, {})
                    and key in glue[t][method]
                    and glue[t][method][key] is not None]
            mean = sum(vals) / len(vals) if vals else None
            row += f"  {fmt(mean, 3):>9}"
        print(row)
    print()

    # ── 7. Training dynamics (SST-2 if available) ─────────────────────────
    section("TABLE 7: TRAINING DYNAMICS — sr(W_eff) per epoch (SST-2)")
    sst2 = glue.get("sst2", {})
    dyn_methods = [m for m in METHOD_ORDER if sst2.get(m, {}).get("per_epoch")]
    if not dyn_methods:
        print("  No per_epoch data found for SST-2.")
    else:
        max_epochs = max(
            len(sst2[m]["per_epoch"]) for m in dyn_methods
        )
        header = f"{'Method':<22}" + "".join(f"  Ep{i+1:>2}" for i in range(max_epochs))
        print(header)
        sep()
        for method in dyn_methods:
            epochs = sst2[method]["per_epoch"]
            row = f"{method:<22}"
            for ep in epochs:
                row += f"  {ep.get('sr_Weff', 0):>5.2f}"
            print(row)
    print()

    # ── 8. Layer profiles ─────────────────────────────────────────────────
    section("TABLE 8: PER-LAYER sr(W_eff) — CoLA and MRPC")
    for task in ["cola", "mrpc"]:
        if task not in glue:
            continue
        print(f"\n  Task: {task.upper()}")
        print(f"  {'Method':<22}" + "".join(f"  L{i:02d}" for i in range(12)))
        sep(width=80)
        for method in ["hybrid_paft", "safe_hybrid_paft", "polar_r8", "full_ft", "pure_paft"]:
            layers = glue.get(task, {}).get(method, {}).get("per_layer", [])
            if not layers:
                continue
            row = f"  {method:<22}"
            for rec in layers[:12]:
                v = rec.get("sr_Weff_final", 0)
                row += f"  {v:>5.2f}"
            print(row)
    print()

    # ── 9. PAFT Stiefel audit ──────────────────────────────────────────────
    if paft:
        section("TABLE 9: STIEFEL INVARIANCE AUDIT — Q drift (should be 0.00e+00)")
        print(f"{'Method':<22}  {'Task':<6}  {'Q_V mean':>10}  {'Q_V max':>10}  {'Q_O mean':>10}  {'Q_O max':>10}")
        sep()
        paft_methods = ["pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"]
        for task in tasks_available:
            for method in paft_methods:
                d = paft.get(task, {}).get(method)
                if d is None:
                    continue
                qv_mean = d.get("Q_V_drift_mean", d.get("Q_drift_mean", "N/A"))
                qv_max  = d.get("Q_V_drift_max",  d.get("Q_drift_max",  "N/A"))
                qo_mean = d.get("Q_O_drift_mean", "N/A")
                qo_max  = d.get("Q_O_drift_max",  "N/A")
                def fe(v):
                    return f"{float(v):.2e}" if v != "N/A" else "  N/A    "
                print(f"{method:<22}  {task:<6}  {fe(qv_mean):>10}  {fe(qv_max):>10}  {fe(qo_mean):>10}  {fe(qo_max):>10}")
        print()

    # ── 9b. S asymmetry audit (hybrid-PAFT micro-rotation diagnosis) ──────────
    section("TABLE 9b: S ASYMMETRY — micro-rotation diagnosis (hybrid variants only)")
    print("  symmetry_ratio = ||S - Sᵀ||_F / ||S||_F")
    print("  < 0.01 → S stayed symmetric, no micro-rotation")
    print("  > 0.10 → real asymmetry, micro-rotation Q' exists")
    print()
    print(f"  {'Method':<22}  {'Task':<6}  {'S_V asym':>10}  {'S_O asym':>10}  {'Verdict':>20}")
    sep()
    hybrid_methods = ["hybrid_paft", "safe_hybrid_paft"]
    if paft:
        for task in tasks_available:
            for method in hybrid_methods:
                d = paft.get(task, {}).get(method)
                if d is None:
                    continue
                sv = d.get("S_V_asymmetry_mean")
                so = d.get("S_O_asymmetry_mean")
                if sv is None:
                    continue
                verdict = "symmetric (no rotation)" if sv < 0.01 else \
                    "slight drift" if sv < 0.05 else \
                        "real micro-rotation"
                print(f"  {method:<22}  {task:<6}  {sv:>10.4f}  {so:>10.4f}  {verdict:>20}")
    print()

    # ── 10b. Nuclear norm ratio ─────────────────────────────────────────────
    # Placed after TABLE 6 since it requires W_init and is computed separately.
    # Values near 1.0 = spectral energy conserved.
    # Values > 1.0 = fine-tuning increased total singular value mass.
    # Values < 1.0 = fine-tuning reduced it.
    # PAFT prediction: stays close to 1.0 (S changes emphasis, not total energy).
    section("TABLE 10: NUCLEAR NORM RATIO ‖W_adapted‖★ / ‖W_0‖★ — per task (Appendix)")
    nn_key = "nuclear_norm_ratio"
    any_nn = any(
        nn_key in glue.get(t, {}).get(m, {})
        for t in tasks_available for m in METHOD_ORDER
    )
    if not any_nn:
        print("  nuclear_norm_ratio not in cache yet.")
        print("  Add isotropy/participation_ratio/nuclear_norm_ratio to build_cache.py")
        print("  and rebuild the cache to populate this table.")
    else:
        header = f"{'Method':<22}" + "".join(f"  {t.upper():<8}" for t in tasks_available) + f"  {'Mean':>7}"
        print(header)
        sep(width=100)
        for method in METHOD_ORDER:
            row = f"{method:<22}"
            vals = []
            for task in tasks_available:
                v = glue.get(task, {}).get(method, {}).get(nn_key)
                row += f"  {fmt(v, 4)}"
                if v is not None:
                    vals.append(v)
            mean = sum(vals) / len(vals) if vals else None
            row += f"  {fmt(mean, 4):>7}"
            print(row)
        print()
        print("  Reference: 1.0000 = no change in spectral energy.")
        print("  > 1.0 = adapter increased total singular value mass.")
        print("  < 1.0 = adapter reduced it.")
        print("  PAFT variants predicted to stay closest to 1.0.")
    print()


    # ── 10. Sanity check flags ─────────────────────────────────────────────
    section("SANITY CHECKS")
    issues = []

    # Check: all methods have data for all tasks
    for task in tasks_available:
        for method in METHOD_ORDER:
            if method not in glue.get(task, {}):
                issues.append(f"  MISSING: {task}/{method} — not in cache")

    # Check: sr values in plausible range
    for task in tasks_available:
        for method, d in glue.get(task, {}).items():
            sr_f = d.get("sr_Weff_final")
            if sr_f is not None and (sr_f > 64.5 or sr_f < 0.5):
                issues.append(f"  SUSPICIOUS sr(W_eff)={sr_f:.2f} for {task}/{method} "
                               f"(expected 1-64 for per-head [64x768] weights)")
            sr_d = d.get("sr_deltaW_V")
            if sr_d is not None and sr_d > 65:
                issues.append(f"  SUSPICIOUS sr(ΔW)={sr_d:.2f} for {task}/{method} "
                               f"(exceeds max possible for per-head weight)")

    # Check: LoRA sr looks different from pretrained
    for task in tasks_available:
        for lm in ["lora_r8", "lora_r64"]:
            d = glue.get(task, {}).get(lm, {})
            sr_f = d.get("sr_Weff_final")
            sr_i = d.get("sr_Weff_init")
            if sr_f is not None and sr_i is not None and abs(sr_f - sr_i) < 0.01:
                issues.append(f"  LORA UNCHANGED: {task}/{lm} sr_final≈sr_init ({sr_f:.4f}) "
                               f"— recovery may have failed")

    if not issues:
        print("  All checks passed.")
    else:
        for iss in issues:
            print(iss)
    print()


if __name__ == "__main__":
    main()

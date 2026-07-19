#!/usr/bin/env python3
"""
generate_paper_outputs.py

Generates all LaTeX tables, figures, and analysis dump for the PAFT paper.
Run from project root: python3 generate_paper_outputs.py

Outputs written to results/analysis/:
    table_glue_performance.tex
    table_llama_performance.tex
    table_stable_rank.tex
    table_all_metrics.tex
    table_sr_per_task.tex
    table_sr_delta_w.tex
    table_training_dynamics.tex
    table_per_layer_cola.tex
    table_per_layer_mrpc.tex
    table_asymmetry.tex
    table_q_drift.tex
    figures/ (all plots)
    analysis_dump.txt
"""

import json
import os
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results/analysis")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Method display names for tables
METHOD_NAMES = {
    "frozen":          "Frozen",
    "bitfit":          "BitFit",
    "svf":             "SVF",
    "pure_paft":       "pure-PAFT (Ours)",
    "hybrid_paft":     "hybrid-PAFT (Ours)",
    "safe_pure_paft":  "safe-pure-PAFT (Ours)",
    "safe_hybrid_paft":"safe-hybrid-PAFT (Ours)",
    "lora_r8":         "LoRA $r{=}8$",
    "lora_r64":        "LoRA $r{=}64$",
    "polar_r8":        "PoLAR $r{=}8$",
    "full_ft":         "Full fine-tuning",
}

# Task display names
TASK_NAMES = {
    "cola": "CoLA", "mrpc": "MRPC", "rte": "RTE",
    "stsb": "STS-B", "sst2": "SST-2", "qnli": "QNLI",
    "mnli": "MNLI", "qqp": "QQP",
}

# Ordered method lists
GLUE_METHODS_ORDERED = [
    "frozen", "bitfit", "svf",
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "lora_r8", "lora_r64", "polar_r8", "full_ft",
]

LLAMA_METHODS_ORDERED = [
    "frozen", "pure_paft", "hybrid_paft", "lora_r8", "polar_r8",
]

GLUE_TASKS_ORDERED = [
    "cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp",
]

# Primary metric per task (for GLUE table)
TASK_METRICS = {
    "cola": "MCC", "mrpc": "F1", "rte": "Acc",
    "stsb": "Pearson", "sst2": "Acc", "qnli": "Acc",
    "mnli": "Acc", "qqp": "F1",
}

# Methods that are PAFT variants (get underline for best PAFT)
PAFT_METHODS = {
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
}

# Methods with no additive update (sr(ΔW) = N/A)
NO_DELTA_W = {"frozen", "bitfit"}


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────

def load_data():
    with open(RESULTS_DIR / "metrics_cache.json") as f:
        cache = json.load(f)["glue"]
    with open(RESULTS_DIR / "llama_results.json") as f:
        llama = json.load(f)
    return cache, llama


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def fmt(val, decimals=4):
    """Format a float to fixed decimals."""
    if val is None:
        return "---"
    return f"{val:.{decimals}f}"


def bold(s):
    return f"\\textbf{{{s}}}"


def uline(s):
    return f"\\underline{{{s}}}"


def bold_uline(s):
    return f"\\textbf{{\\underline{{{s}}}}}"


def find_best(scores, methods, paft_set):
    """
    Returns (best_overall_method, best_paft_method) by score.
    Ignores None values.
    """
    valid = {m: s for m, s in zip(methods, scores) if s is not None}
    if not valid:
        return None, None
    best_overall = max(valid, key=valid.get)
    paft_valid = {m: s for m, s in valid.items() if m in paft_set}
    best_paft = max(paft_valid, key=paft_valid.get) if paft_valid else None
    return best_overall, best_paft


def apply_formatting(val_str, method, best_overall, best_paft):
    """Apply bold/underline formatting."""
    is_best = (method == best_overall)
    is_best_paft = (method == best_paft)
    if is_best and is_best_paft:
        return bold_uline(val_str)
    elif is_best:
        return bold(val_str)
    elif is_best_paft:
        return uline(val_str)
    return val_str

# ─────────────────────────────────────────────────────────────
# Table 1: GLUE Performance
# ─────────────────────────────────────────────────────────────
# REMOVED: make_glue_table() used to live here, duplicating
# table_syntax_semantics.py under the same \label{tab:glue_performance}.
# That duplication was the actual source of the "multiply defined labels"
# problem found earlier. table_syntax_semantics.py is now the single
# source of truth for this table — run it directly:
#   python3 -m analysis.tables.table_syntax_semantics

# ─────────────────────────────────────────────────────────────
# Table 2: LLaMA Performance
# ─────────────────────────────────────────────────────────────

def make_llama_table(llama):
    """Generate table_llama_performance.tex"""

    tasks = ["boolq", "hellaswag", "arc_challenge"]
    task_display = {
        "boolq": "BoolQ",
        "hellaswag": "HellaSwag",
        "arc_challenge": "ARC-C",
    }
    methods = LLAMA_METHODS_ORDERED
    results = llama["results"]

    # Collect scores
    scores = {}
    for method in methods:
        scores[method] = {}
        for task in tasks:
            val = results.get(task, {}).get(method, {}).get("accuracy")
            scores[method][task] = val

    # Find best per task
    best_overall = {}
    best_paft = {}
    paft_llama = {"pure_paft", "hybrid_paft"}
    for task in tasks:
        task_scores = [scores[m][task] for m in methods]
        bo, bp = find_best(task_scores, methods, paft_llama)
        best_overall[task] = bo
        best_paft[task] = bp

    # Mean per method (only over available tasks)
    means = {}
    for method in methods:
        vals = [v for v in scores[method].values() if v is not None]
        means[method] = np.mean(vals) if vals else None

    task_headers = " & ".join(
        f"\\textbf{{{task_display[t]}}}" for t in tasks
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{LLaMA-3.2-3B commonsense reasoning results (accuracy). "
        r"\textbf{Bold}: best overall per column. "
        r"\underline{Underline}: best PAFT variant. "
        r"LoRA targets both $\mathbf{W}_{V,h}$ and $\mathbf{W}_{O,h}$; "
        r"all other methods target $\mathbf{W}_{V,h}$ only "
        r"(Section~\ref{subsec:llama_setup}).}"
    )
    lines.append(r"\label{tab:llama_performance}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        f"\\textbf{{Method}} & {task_headers} & \\textbf{{Mean}} \\\\"
    )
    lines.append(r"\midrule")

    groups = [
        ["frozen"],
        ["pure_paft", "hybrid_paft"],
        ["lora_r8", "polar_r8"],
    ]

    for g_idx, group in enumerate(groups):
        for method in group:
            name = METHOD_NAMES[method]
            cells = []
            for task in tasks:
                val = scores[method][task]
                val_str = fmt(val, 4) if val is not None else "---"
                val_str = apply_formatting(
                    val_str, method, best_overall[task], best_paft[task]
                )
                cells.append(val_str)
            mean_val = means[method]
            mean_str = fmt(mean_val, 4) if mean_val is not None else "---"
            row = f"{name} & " + " & ".join(cells) + f" & {mean_str} \\\\"
            lines.append(row)
        if g_idx < len(groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "\n".join(lines)
    path = RESULTS_DIR / "table_llama_performance.tex"
    path.write_text(out)
    print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────
# Table 3: Stable Rank Init vs Final
# ─────────────────────────────────────────────────────────────
# REMOVED: make_stable_rank_table() used to live here, duplicating
# table_stable_rank.py under the same \label{tab:stable_rank}, but with a
# DIFFERENT column layout (4 columns, %-based delta, hardcoded pretrained
# value 34.745) — this was confirmed to be the actual source of the
# confusing mismatched table_stable_rank.tex output seen earlier in this
# project. table_stable_rank.py is now the single source of truth:
#   python3 -m analysis.tables.table_stable_rank

# ─────────────────────────────────────────────────────────────
# Table 4: Multi-Metric Spectral Profile
# ─────────────────────────────────────────────────────────────
# REMOVED: make_all_metrics_table() used to live here, duplicating
# table_all_metrics.py under the same \label{tab:all_metrics}. This one was
# worse than a benign duplicate: it had its own hardcoded fallback values
# (effective_rank default 63.678, frozen_sr=34.745, spectral_entropy default
# 4.154 — the exact anti-pattern eliminated from build_cache.py earlier in
# this project) and computed Isotropy as a direct 1/kappa reciprocal, which
# is no longer a valid identity once kappa and Isotropy are independently
# per-head-averaged (kappa is an outlier-sensitive arithmetic mean; Isotropy
# is not). table_all_metrics.py is now the single source of truth:
#   python3 -m analysis.tables.table_all_metrics

# ─────────────────────────────────────────────────────────────
# Table 5: Training Dynamics
# ─────────────────────────────────────────────────────────────
def make_training_dynamics_table(cache):
    """Generate table_training_dynamics.tex — sr(W_eff) per epoch on SST-2"""

    # Methods to include — now LoRA and PoLAR have real merged values
    methods = [
        "frozen", "bitfit", "svf",
        "pure_paft", "hybrid_paft",
        "safe_pure_paft", "safe_hybrid_paft",
        "lora_r8", "lora_r64", "polar_r8", "full_ft",
    ]

    # SST-2 has 5 epochs for most methods
    n_epochs = 5

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{$\operatorname{sr}(\mathbf{W}_{\text{eff}})$ per "
        r"training epoch on SST-2. "
        r"LoRA values computed by merging adapter "
        r"weights with the frozen base model at each checkpoint.}"
    )
    lines.append(r"\label{tab:training_dynamics}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & \textbf{Ep 1} & \textbf{Ep 2} & "
        r"\textbf{Ep 3} & \textbf{Ep 4} & \textbf{Ep 5} \\"
    )
    lines.append(r"\midrule")

    groups = [
        ["frozen", "bitfit"],
        ["svf", "pure_paft", "hybrid_paft",
         "safe_pure_paft", "safe_hybrid_paft"],
        ["lora_r8", "lora_r64", "polar_r8"],
        ["full_ft"],
    ]

    for g_idx, group in enumerate(groups):
        for method in group:
            entry = cache.get("sst2", {}).get(method, {})
            per_epoch_raw = entry.get("per_epoch", [])

            # Build epoch → sr mapping regardless of format
            epoch_sr = {}
            if isinstance(per_epoch_raw, list):
                # Format: [{"epoch": 1, "sr_Weff": 34.7}, ...]
                for item in per_epoch_raw:
                    if isinstance(item, dict):
                        ep  = item.get("epoch")
                        sr  = item.get("sr_Weff")
                        if ep is not None and sr is not None:
                            epoch_sr[int(ep)] = sr
            elif isinstance(per_epoch_raw, dict):
                # Format: {"1": {"sr_Weff": 34.7}, ...}
                for k, v in per_epoch_raw.items():
                    sr = v.get("sr_Weff") if isinstance(v, dict) else v
                    if sr is not None:
                        epoch_sr[int(k)] = sr

            epoch_vals = []
            for ep in range(1, n_epochs + 1):
                val = epoch_sr.get(ep)
                epoch_vals.append(
                    fmt(val, 2) if val is not None else "---"
                )

            name = METHOD_NAMES[method]
            row = f"{name} & " + " & ".join(epoch_vals) + r" \\"
            lines.append(row)

        if g_idx < len(groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "\n".join(lines)
    path = RESULTS_DIR / "table_training_dynamics.tex"
    path.write_text(out)
    print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────
# Table 6: Per-Layer sr(W_eff)
# ─────────────────────────────────────────────────────────────
def make_per_layer_tables(cache):
    """Generate table_per_layer_cola.tex and table_per_layer_mrpc.tex"""

    layer_methods = [
        "pure_paft", "hybrid_paft",
        "safe_hybrid_paft", "polar_r8", "full_ft",
    ]

    for task in ["cola", "mrpc"]:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(
            f"\\caption{{Final $\\operatorname{{sr}}(\\mathbf{{W}}_{{"
            f"\\text{{eff}}}})$ at each encoder layer for "
            f"\\textsc{{{TASK_NAMES[task]}}}.}}"
        )
        lines.append(f"\\label{{tab:per_layer_{task}}}")
        lines.append(r"\begin{tabular}{l" + "c" * 12 + "}")
        lines.append(r"\toprule")
        layer_headers = " & ".join(
            f"\\textbf{{L{i:02d}}}" for i in range(12)
        )
        lines.append(f"\\textbf{{Method}} & {layer_headers} \\\\")
        lines.append(r"\midrule")

        for method in layer_methods:
            entry = cache.get(task, {}).get(method, {})
            per_layer_raw = entry.get("per_layer", [])
            name = METHOD_NAMES[method]

            # Build layer_index → sr_Weff_final mapping
            layer_sr = {}
            if isinstance(per_layer_raw, list):
                for item in per_layer_raw:
                    if isinstance(item, dict):
                        idx = item.get("layer")
                        sr  = item.get("sr_Weff_final")
                        if idx is not None and sr is not None:
                            layer_sr[int(idx)] = sr
            elif isinstance(per_layer_raw, dict):
                for k, v in per_layer_raw.items():
                    sr = v.get("sr_Weff") if isinstance(v, dict) else v
                    if sr is not None:
                        layer_sr[int(k)] = sr

            layer_vals = []
            for i in range(12):
                val = layer_sr.get(i)
                layer_vals.append(
                    fmt(val, 2) if val is not None else "---"
                )

            row = f"{name} & " + " & ".join(layer_vals) + r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        out = "\n".join(lines)
        path = RESULTS_DIR / f"table_per_layer_{task}.tex"
        path.write_text(out)
        print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────
# Table 7: sr(W_eff) per task (appendix)
# ─────────────────────────────────────────────────────────────

def make_sr_per_task_table(cache):
    """Generate table_sr_per_task.tex"""

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Final $\operatorname{sr}(\mathbf{W}_{\text{eff}})$ "
        r"for every method and GLUE task. "
        r"Lower values indicate greater geometric damage. "
        r"All methods share the same pretrained baseline.}"
    )
    lines.append(r"\label{tab:sr_per_task}")

    task_headers = " & ".join(
        f"\\textbf{{{TASK_NAMES[t]}}}" for t in GLUE_TASKS_ORDERED
    )
    lines.append(r"\begin{tabular}{l" + "c" * 8 + "}")
    lines.append(r"\toprule")
    lines.append(f"\\textbf{{Method}} & {task_headers} \\\\")
    lines.append(r"\midrule")

    groups = [
        ["frozen", "bitfit"],
        ["svf", "pure_paft", "hybrid_paft",
         "safe_pure_paft", "safe_hybrid_paft"],
        ["lora_r8", "lora_r64", "polar_r8"],
        ["full_ft"],
    ]

    for g_idx, group in enumerate(groups):
        for method in group:
            name = METHOD_NAMES[method]
            cells = []
            for task in GLUE_TASKS_ORDERED:
                val = cache.get(task, {}).get(method, {}).get("sr_Weff_final")
                cells.append(fmt(val, 3) if val is not None else "---")
            row = f"{name} & " + " & ".join(cells) + r" \\"
            lines.append(row)
        if g_idx < len(groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "\n".join(lines)
    path = RESULTS_DIR / "table_sr_per_task.tex"
    path.write_text(out)
    print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────
# Table 8: sr(ΔW) per task (appendix)
# ─────────────────────────────────────────────────────────────

def make_sr_delta_w_table(cache):
    """Generate table_sr_delta_w.tex"""

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Stable rank of the weight update "
        r"$\operatorname{sr}(\Delta\mathbf{W}_V)$ per GLUE task. "
        r"N/A: method does not produce an additive update.}"
    )
    lines.append(r"\label{tab:sr_delta_w}")

    task_headers = " & ".join(
        f"\\textbf{{{TASK_NAMES[t]}}}" for t in GLUE_TASKS_ORDERED
    )
    lines.append(r"\begin{tabular}{l" + "c" * 8 + "}")
    lines.append(r"\toprule")
    lines.append(f"\\textbf{{Method}} & {task_headers} \\\\")
    lines.append(r"\midrule")

    groups = [
        ["frozen", "bitfit"],
        ["svf", "pure_paft", "hybrid_paft",
         "safe_pure_paft", "safe_hybrid_paft"],
        ["lora_r8", "lora_r64", "polar_r8"],
        ["full_ft"],
    ]

    for g_idx, group in enumerate(groups):
        for method in group:
            name = METHOD_NAMES[method]
            cells = []
            for task in GLUE_TASKS_ORDERED:
                if method in NO_DELTA_W:
                    cells.append("N/A")
                else:
                    val = cache.get(task, {}).get(
                        method, {}
                    ).get("sr_deltaW_V")
                    cells.append(fmt(val, 2) if val is not None else "---")
            row = f"{name} & " + " & ".join(cells) + r" \\"
            lines.append(row)
        if g_idx < len(groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "\n".join(lines)
    path = RESULTS_DIR / "table_sr_delta_w.tex"
    path.write_text(out)
    print(f"Written: {path}")

# ─────────────────────────────────────────────────────────────
# Table: Q Drift Audit
# ─────────────────────────────────────────────────────────────

def make_q_drift_table(paft_cache):
    """Generate table_q_drift.tex"""

    methods = [
        "pure_paft", "hybrid_paft",
        "safe_pure_paft", "safe_hybrid_paft",
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Frobenius drift $\|\mathbf{Q}_{\text{final}} - "
        r"\mathbf{Q}_{\text{init}}\|_F$ for all PAFT variants across "
        r"all GLUE tasks and both projections.}"
    )
    lines.append(r"\label{tab:q_drift}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & "
        r"$\|\Delta\mathbf{Q}_V\|_F$ \textbf{Mean} & "
        r"$\|\Delta\mathbf{Q}_V\|_F$ \textbf{Max} & "
        r"$\|\Delta\mathbf{Q}_O\|_F$ \textbf{Mean} & "
        r"$\|\Delta\mathbf{Q}_O\|_F$ \textbf{Max} \\"
    )
    lines.append(r"\midrule")

    tasks = list(paft_cache.keys())

    for method in methods:
        # Average drift across all tasks
        v_mean_vals, v_max_vals = [], []
        o_mean_vals, o_max_vals = [], []

        for task in tasks:
            entry = paft_cache.get(task, {}).get(method, {})
            if not entry:
                continue
            v_mean_vals.append(entry.get("Q_V_drift_mean", 0.0))
            v_max_vals.append(entry.get("Q_V_drift_max", 0.0))
            o_mean_vals.append(entry.get("Q_O_drift_mean", 0.0))
            o_max_vals.append(entry.get("Q_O_drift_max", 0.0))

        if not v_mean_vals:
            continue

        v_mean = np.mean(v_mean_vals)
        v_max  = np.max(v_max_vals)
        o_mean = np.mean(o_mean_vals)
        o_max  = np.max(o_max_vals)

        name = METHOD_NAMES[method]

        # Format as scientific notation if zero
        def fmt_drift(val):
            if val == 0.0:
                return r"$0.00\text{e}{+}00$"
            return f"{val:.2e}"

        row = (
            f"{name} & "
            f"{fmt_drift(v_mean)} & "
            f"{fmt_drift(v_max)} & "
            f"{fmt_drift(o_mean)} & "
            f"{fmt_drift(o_max)} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "\n".join(lines)
    path = RESULTS_DIR / "table_q_drift.tex"
    path.write_text(out)
    print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────
# Table: S Asymmetry and Micro-Rotation
# ─────────────────────────────────────────────────────────────
def make_asymmetry_table(paft_cache):
    """Generate table_asymmetry.tex with gradient steps column."""

    # Gradient steps per task: floor(N_train / batch_size) * epochs
    # batch_size=32, no gradient accumulation.
    # Epochs: 10 (CoLA/MRPC/RTE/STS-B), 5 (SST-2/QNLI), 3 (MNLI/QQP)
    # Dataset sizes from Wang et al. (2018) official GLUE splits.
    # Verified against results/glue/{task}/{method}/config.json.
    TASK_STEPS = {
        "rte":   7_780,
        "mrpc": 11_460,
        "stsb": 17_960,
        "cola": 26_720,
        "sst2": 10_520,
        "qnli": 16_365,
        "qqp":  34_110,
        "mnli": 36_816,
    }

    # Task metadata: (key, display, N_train, grad_steps)
    tasks_ordered = [
        ("rte",  "RTE",   2_490,   TASK_STEPS["rte"]),
        ("mrpc", "MRPC",  3_668,   TASK_STEPS["mrpc"]),
        ("stsb", "STS-B", 5_749,   TASK_STEPS["stsb"]),
        ("cola", "CoLA",  8_551,   TASK_STEPS["cola"]),
        ("sst2", "SST-2", 67_349,  TASK_STEPS["sst2"]),
        ("qnli", "QNLI",  104_743, TASK_STEPS["qnli"]),
        ("qqp",  "QQP",   363_849, TASK_STEPS["qqp"]),
        ("mnli", "MNLI",  392_702, TASK_STEPS["mnli"]),
    ]

    # Sort by gradient steps ascending
    tasks_ordered = sorted(tasks_ordered, key=lambda x: x[2])

    methods = ["hybrid_paft", "safe_hybrid_paft"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption{Symmetry ratio "
        r"$\|\mathbf{M} - \mathbf{M}^\top\|_F / \|\mathbf{M}\|_F$ "
        r"for hybrid-PAFT variants, ordered by training set size $N$. "
        r"Values above $0.05$ indicate meaningful asymmetry, "
        r"corresponding to a learned micro-rotation $\mathbf{Q}'$ "
        r"within the per-head subspace.}"
    )
    lines.append(r"\label{tab:asymmetry}")
    lines.append(r"\begin{tabular}{lrrcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Task} & "
        r"$N$ & "
        r"\textbf{Steps} & "
        r"\multicolumn{2}{c}{\textbf{hybrid-PAFT}} & "
        r"\multicolumn{2}{c}{\textbf{safe-hybrid-PAFT}} \\"
    )
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r" & & & $M_V$ & $M_O$ & $M_V$ & $M_O$ \\")
    lines.append(r"\midrule")

    for task_key, task_display, n, steps in tasks_ordered:
        cells = [
            f"\\textsc{{{task_display}}}",
            f"{n:,}",
            f"{steps:,}",
        ]
        for method in methods:
            entry = paft_cache.get(task_key, {}).get(method, {})
            sv = entry.get("S_V_asymmetry_mean")
            so = entry.get("S_O_asymmetry_mean")
            cells.append(fmt(sv, 4) if sv is not None else "---")
            cells.append(fmt(so, 4) if so is not None else "---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "\n".join(lines)
    path = RESULTS_DIR / "table_asymmetry.tex"
    path.write_text(out)
    print(f"Written: {path}")
# ─────────────────────────────────────────────────────────────
# Table: LLaMA Geometric Analysis
# ─────────────────────────────────────────────────────────────
def make_llama_geometric_table():
    """Generate table_llama_geometric.tex"""
    import torch

    LLAMA_DIR = Path("results/llama")
    # Main paper: BoolQ (2 epochs) and HellaSwag (1 epoch)
    # ARC-Challenge goes to appendix — minimal adaptation
    tasks_main    = ["boolq", "hellaswag"]
    tasks_all     = ["boolq", "hellaswag", "arc_challenge"]
    methods       = ["frozen", "pure_paft", "hybrid_paft",
                     "lora_r8", "polar_r8"]

    task_display  = {
        "boolq":         "BoolQ",
        "hellaswag":     "HellaSwag",
        "arc_challenge": "ARC-C",
    }

    def load_wv_metrics(task: str, method: str) -> dict | None:
        """Load W_V global metrics, using merged file for LoRA."""
        if method == "lora_r8":
            path = (LLAMA_DIR / task / method /
                    "final" / "geometric_health_merged.pt")
        else:
            path = (LLAMA_DIR / task / method /
                    "final" / "geometric_health.pt")

        if not path.exists():
            return None
        try:
            data = torch.load(path, map_location="cpu",
                              weights_only=True)
            g  = data.get("global", {})
            wv = g.get("W_V", {})
            return wv if wv else None
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")
            return None

    # Collect results
    results = {}
    for method in methods:
        results[method] = {}
        for task in tasks_all:
            wv = load_wv_metrics(task, method)
            if wv:
                results[method][task] = {
                    "sr":   wv.get("V_stable_rank"),
                    "cond": wv.get("V_condition_number"),
                    "iso":  wv.get("V_isotropy"),
                }
            else:
                results[method][task] = None

    # ── Main paper table (BoolQ + HellaSwag) ──────────────────
    def build_table(tasks: list, label: str, caption: str) -> str:
        # Get pretrained baseline from frozen init
        init_wv = {}
        for task in tasks:
            path = (LLAMA_DIR / task / "frozen" /
                    "init" / "geometric_health.pt")
            if path.exists():
                data = torch.load(path, map_location="cpu",
                                  weights_only=True)
                wv = data.get("global", {}).get("W_V", {})
                init_wv[task] = wv

        n_tasks  = len(tasks)
        col_spec = "l" + "cc" * n_tasks  # sr + cond per task

        task_headers = " & ".join(
            f"\\multicolumn{{2}}{{c}}"
            f"{{\\textbf{{{task_display[t]}}}}}"
            for t in tasks
        )
        sub_headers = " & ".join(
            "$\\operatorname{sr}$ & $\\kappa$"
            for _ in tasks
        )
        # cmidrule per task
        cmidrules = " ".join(
            f"\\cmidrule(lr){{{2+i*2}-{3+i*2}}}"
            for i in range(n_tasks)
        )

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        lines.append(f"\\textbf{{Method}} & {task_headers} \\\\")
        lines.append(cmidrules)
        lines.append(f" & {sub_headers} \\\\")
        lines.append(r"\midrule")

        # Pretrained row
        pretrained_cells = []
        for task in tasks:
            wv = init_wv.get(task, {})
            sr   = wv.get("V_stable_rank")
            cond = wv.get("V_condition_number")
            pretrained_cells.append(
                f"{fmt(sr, 2)} & {fmt(cond, 2)}"
                if sr else "--- & ---"
            )
        lines.append(
            "Pretrained & " +
            " & ".join(pretrained_cells) + r" \\"
        )
        lines.append(r"\midrule")

        groups = [
            ["frozen"],
            ["pure_paft", "hybrid_paft"],
            ["lora_r8", "polar_r8"],
        ]

        for g_idx, group in enumerate(groups):
            for method in group:
                name  = METHOD_NAMES[method]
                cells = []
                for task in tasks:
                    entry = results[method].get(task)
                    if entry:
                        sr   = entry.get("sr")
                        cond = entry.get("cond")
                        cells.append(
                            f"{fmt(sr, 2)} & {fmt(cond, 2)}"
                            if sr else "--- & ---"
                        )
                    else:
                        cells.append("--- & ---")
                row = f"{name} & " + " & ".join(cells) + r" \\"
                lines.append(row)
            if g_idx < len(groups) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    # Main paper table
    # NOTE for prose (this caption is no longer auto-generated into the
    main_caption = (
        r"Geometric health of $\mathbf{W}_{V,h}$ for LLaMA-3.2-3B "
        r"after fine-tuning. $\operatorname{sr}$: stable rank. "
        r"$\kappa$: condition number ($\sigma_{\max}/\sigma_{\min}$). "
        r"Pretrained: $\operatorname{sr} \approx 68.9$, "
        r"$\kappa \approx 2.43$. "
        r"HellaSwag trained for 1 epoch; BoolQ for 2 epochs. "
        r"LoRA values use merged $\mathbf{W}_{\text{eff}}$."
    )
    main_tex = build_table(
        tasks_main,
        label="tab:llama_geometric",
        caption=main_caption,
    )
    path_main = RESULTS_DIR / "table_llama_geometric.tex"
    path_main.write_text(main_tex)
    print(f"Written: {path_main}")

    # Appendix table (all three tasks)
    app_caption = (
        r"Full geometric health of $\mathbf{W}_{V,h}$ for "
        r"LLaMA-3.2-3B across all three commonsense tasks "
        r"(ARC-Challenge: $N{=}1{,}119$). "
        r"See Table~\ref{tab:llama_geometric} for BoolQ and "
        r"HellaSwag results."
    )
    app_tex = build_table(
        tasks_all,
        label="tab:llama_geometric_full",
        caption=app_caption,
    )
    path_app = RESULTS_DIR / "table_llama_geometric_appendix.tex"
    path_app.write_text(app_tex)
    print(f"Written: {path_app}")
# ─────────────────────────────────────────────────────────────
# Analysis Dump
# ─────────────────────────────────────────────────────────────

def make_analysis_dump(cache, llama, paft_cache=None):
    """Write clean analysis_dump.txt with all key numbers."""

    lines = []
    sep = "═" * 80

    lines.append(sep)
    lines.append("  PAFT PAPER — ANALYSIS DUMP")
    lines.append(sep)

    # GLUE scores
    lines.append("\nTABLE 1: GLUE TASK PERFORMANCE")
    lines.append("─" * 80)
    header = f"{'Method':<24}" + "".join(
        f"{TASK_NAMES[t]:>8}" for t in GLUE_TASKS_ORDERED
    ) + f"{'Mean':>8}"
    lines.append(header)
    lines.append("─" * 80)

    for method in GLUE_METHODS_ORDERED:
        vals = []
        for task in GLUE_TASKS_ORDERED:
            v = cache.get(task, {}).get(method, {}).get("task_score")
            vals.append(v)
        mean = np.mean([v for v in vals if v is not None])
        row = f"{METHOD_NAMES[method]:<24}"
        for v in vals:
            row += f"{v:8.4f}" if v is not None else "     ---"
        row += f"{mean:8.4f}"
        lines.append(row)

    # Stable rank
    lines.append(f"\n\nTABLE 2: STABLE RANK INIT vs FINAL (task average)")
    lines.append("─" * 80)
    lines.append(
        f"{'Method':<24}{'sr_init':>10}{'sr_final':>10}"
        f"{'delta':>10}{'delta%':>10}"
    )
    lines.append("─" * 80)

    for method in GLUE_METHODS_ORDERED:
        sr_inits, sr_finals = [], []
        for task in GLUE_TASKS_ORDERED:
            entry = cache.get(task, {}).get(method, {})
            if entry.get("sr_Weff_init"):
                sr_inits.append(entry["sr_Weff_init"])
            if entry.get("sr_Weff_final"):
                sr_finals.append(entry["sr_Weff_final"])
        if not sr_inits:
            continue
        sr_i = np.mean(sr_inits)
        sr_f = np.mean(sr_finals)
        delta = sr_f - sr_i
        pct = delta / sr_i * 100
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{METHOD_NAMES[method]:<24}{sr_i:10.3f}{sr_f:10.3f}"
            f"{sign+f'{delta:.3f}':>10}{sign+f'{pct:.2f}%':>10}"
        )

    # LLaMA results
    lines.append(f"\n\nTABLE 3: LLAMA COMMONSENSE RESULTS")
    lines.append("─" * 80)
    lines.append(
        f"{'Method':<24}{'BoolQ':>10}{'HellaSwag':>12}"
        f"{'ARC-C':>10}{'Mean':>10}"
    )
    lines.append("─" * 80)

    llama_tasks = ["boolq", "hellaswag", "arc_challenge"]
    for method in LLAMA_METHODS_ORDERED:
        vals = []
        for task in llama_tasks:
            v = llama["results"].get(task, {}).get(method, {}).get("accuracy")
            vals.append(v)
        mean = np.mean([v for v in vals if v is not None])
        row = f"{METHOD_NAMES[method]:<24}"
        for v in vals:
            row += f"{v:10.4f}" if v is not None else "       ---"
        row += f"{mean:10.4f}"
        lines.append(row)

    # Key numbers
    lines.append(f"\n\nKEY NUMBERS FOR PAPER")
    lines.append("─" * 80)

    # Q drift — computed from paft_cache, same source table_q_drift.tex uses,
    # instead of asserted. Do not assume it's exactly zero; report what the
    # data actually says.
    paft_methods = ["pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"]
    if paft_cache:
        all_drift_vals = []
        for task_data in paft_cache.values():
            for method in paft_methods:
                entry = task_data.get(method, {})
                for key in ("Q_V_drift_max", "Q_O_drift_max"):
                    v = entry.get(key)
                    if v is not None:
                        all_drift_vals.append(v)
        if all_drift_vals:
            max_drift = max(all_drift_vals)
            lines.append(f"Q drift (max over all PAFT variants, tasks, projections): {max_drift:.2e}")
            if max_drift < 1e-8:
                lines.append("(Consistent with exact invariance: frozen Q buffer receives no gradient by construction)")
            else:
                lines.append("(NONZERO — does not match the 'exact invariance' claim; check before using in the paper)")
        else:
            lines.append("Q drift: no Q_V_drift_max/Q_O_drift_max entries found in paft_cache.json")
    else:
        lines.append("Q drift: paft_cache.json not loaded — cannot compute")

    # sr correlation — real Pearson r per task (mirrors table_correlation.py),
    # not a placeholder pointer to "run it separately."
    lines.append("\nPearson r between sr(W_eff) and task score (per task, methods excluding frozen):")
    try:
        from scipy.stats import pearsonr
        task_rs = []
        for task in GLUE_TASKS_ORDERED:
            task_data = cache.get(task, {})
            scores_, srs_ = [], []
            for method, entry in task_data.items():
                if "frozen" in method:
                    continue
                s = entry.get("task_score")
                sr = entry.get("sr_Weff_final")
                if s is not None and sr is not None:
                    scores_.append(s)
                    srs_.append(sr)
            if len(scores_) < 3:
                continue
            r_val, p_val = pearsonr(srs_, scores_)
            task_rs.append(r_val)
            lines.append(f"  {TASK_NAMES.get(task, task):<8} r={r_val:+.4f}  p={p_val:.2e}  (n={len(scores_)})")
        if task_rs:
            lines.append(f"  Mean across tasks: r={np.mean(task_rs):+.4f}")
        else:
            lines.append("  Insufficient data to compute correlation for any task.")
    except ImportError:
        lines.append("  [SKIP] scipy not available — install scipy to compute this")

    out = "\n".join(lines)
    path = RESULTS_DIR / "analysis_dump.txt"
    path.write_text(out)
    print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    cache, llama = load_data()

    # Load paft cache for Q drift and asymmetry tables
    paft_cache_path = RESULTS_DIR / "paft_cache.json"
    if paft_cache_path.exists():
        with open(paft_cache_path) as f:
            paft_cache = json.load(f)
    else:
        print("Warning: paft_cache.json not found — "
              "Q drift and asymmetry tables will be skipped")
        paft_cache = {}

    print("\nGenerating tables...")
    # make_glue_table, make_stable_rank_table, make_all_metrics_table
    # removed — see comments above their old locations. Run those tables
    # via the standalone scripts instead:
    #   python3 -m analysis.tables.table_syntax_semantics
    #   python3 -m analysis.tables.table_stable_rank
    #   python3 -m analysis.tables.table_all_metrics
    #   python3 -m analysis.tables.table_wO_metrics
    make_llama_table(llama)
    make_training_dynamics_table(cache)
    make_per_layer_tables(cache)
    make_sr_per_task_table(cache)
    make_sr_delta_w_table(cache)
    make_q_drift_table(paft_cache)
    make_asymmetry_table(paft_cache)
    make_llama_geometric_table()
    make_analysis_dump(cache, llama, paft_cache)

    print("\nAll outputs generated successfully.")
    print(f"Tables: {RESULTS_DIR}/table_*.tex")
    print(f"Dump:   {RESULTS_DIR}/analysis_dump.txt")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
r"""
analysis/table_syntax_semantics.py

Performance of PAFT variants on syntactic (CoLA) vs semantic tasks.
Reads from metrics_cache.json — consistent with all other table scripts.

Outputs:
  results/analysis/table_syntax_semantics.tex  — complete \input-ready table
  results/analysis/table_syntax_semantics.json — machine-readable cache
"""
import json
import sys
from pathlib import Path


from analysis.plot_style import METHOD_LABELS

OUT_DIR = Path("results/analysis")

# Tasks to show and their display names
SYNTAX_TASKS   = ["cola"]
SEMANTIC_TASKS = ["mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"]

TASK_LABELS = {
    "cola": "CoLA",  "mrpc": "MRPC",  "rte": "RTE",   "stsb": "STS-B",
    "sst2": "SST-2", "qnli": "QNLI",  "mnli": "MNLI", "qqp":  "QQP",
}

TASK_METRIC_LABEL = {
    "cola": "MCC",  "mrpc": "F1",    "rte": "Acc",  "stsb": "Pearson",
    "sst2": "Acc",  "qnli": "Acc",   "mnli": "Acc", "qqp":  "F1",
}

METHOD_ORDER = [
    "frozen",
    "bitfit", "svf",
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "lora_r8", "lora_r64", "polar_r8",
    "full_ft",
]

# Groups for \midrule separation
GROUPS = [
    ["frozen"],
    ["bitfit", "svf"],
    ["pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"],
    ["lora_r8", "lora_r64", "polar_r8"],
    ["full_ft"],
]


def fmt(val, decimals=1) -> str:
    """Format score as percentage ×100 with given decimal places."""
    if val is None:
        return "---"
    return f"{float(val) * 100:.{decimals}f}"


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/metrics_cache.json not found. "
                 "Run build_cache.py first.")

    with open(cache_path) as f:
        glue = json.load(f)["glue"]

    all_tasks = SYNTAX_TASKS + SEMANTIC_TASKS
    available = [t for t in all_tasks if t in glue]

    # ── Terminal preview ───────────────────────────────────────────────────────
    header = f"{'Method':<24}" + "".join(f"  {TASK_LABELS[t]:<7}" for t in available)
    print("\n" + header)
    print("─" * len(header))
    for method in METHOD_ORDER:
        if not any(method in glue.get(t, {}) for t in available):
            continue
        row = f"{METHOD_LABELS.get(method, method):<24}"
        for task in available:
            val = glue.get(task, {}).get(method, {}).get("task_score")
            row += f"  {fmt(val):<7}"
        print(row)

    # ── JSON cache ────────────────────────────────────────────────────────────
    json_data = {}
    for task in available:
        json_data[task] = {
            "type":   "Syntactic" if task in SYNTAX_TASKS else "Semantic",
            "metric": TASK_METRIC_LABEL[task],
        }
        for method in METHOD_ORDER:
            val = glue.get(task, {}).get(method, {}).get("task_score")
            if val is not None:
                json_data[task][method] = float(val)

    # ── LaTeX — complete booktabs table, \input-ready ─────────────────────────
    n_task_cols = len(available)
    col_spec    = "l" + "c" * n_task_cols

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{GLUE task performance (\%). "
        r"CoLA: Matthews correlation; STS-B: Pearson; MRPC/QQP: F1; others: accuracy. "
        r"\textbf{Bold}: best overall. \underline{Underline}: best PAFT variant.}"
    )
    lines.append(r"\label{tab:glue_performance}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row: task names, with type labels below
    task_header = "\\textbf{Method} & " + " & ".join(
        f"\\textbf{{{TASK_LABELS[t]}}}" for t in available
    ) + r" \\"
    lines.append(task_header)

    # Second header row: metric names
    metric_row = " & " + " & ".join(
        f"\\scriptsize {TASK_METRIC_LABEL[t]}" for t in available
    ) + r" \\"
    lines.append(metric_row)
    lines.append(r"\midrule")

    # Data rows with group separators
    for g_idx, group in enumerate(GROUPS):
        for method in group:
            if not any(method in glue.get(t, {}) for t in available):
                continue
            label = METHOD_LABELS.get(method, method)
            # Escape & in label (none expected, but safe)
            cells = []
            for task in available:
                val = glue.get(task, {}).get(method, {}).get("task_score")
                cells.append(fmt(val) if val is not None else "---")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        # Separator between groups, but not after the last
        if g_idx < len(GROUPS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "table_syntax_semantics.json", "w") as f:
        json.dump(json_data, f, indent=2)

    with open(OUT_DIR / "table_syntax_semantics.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {OUT_DIR}/table_syntax_semantics.tex")
    print(f"Saved: {OUT_DIR}/table_syntax_semantics.json")


if __name__ == "__main__":
    main()
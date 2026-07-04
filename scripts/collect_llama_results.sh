#!/bin/bash
# ============================================================================
# scripts/collect_llama_results.sh
#
# Collect and display LLaMA-3.2-3B commonsense reasoning results.
#
# Reads results/llama/<task>/<method>/final/metrics.json for every
# task/method combination, prints a human-readable accuracy table,
# writes a LaTeX table fragment, and saves a JSON summary.
#
# Usage:
#   bash scripts/collect_llama_results.sh
#   bash scripts/collect_llama_results.sh --results_dir results/llama
#   bash scripts/collect_llama_results.sh --output_dir  results/analysis
#
# Output files (in --output_dir):
#   llama_results_summary.txt   Human-readable table
#   llama_results_table.tex     LaTeX tabular fragment for the paper
#   llama_results.json          Structured JSON for downstream scripts
# ============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RESULTS_DIR="results/llama"
OUTPUT_DIR="results/analysis"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results_dir=*) RESULTS_DIR="${1#*=}"; shift ;;
        --output_dir=*)  OUTPUT_DIR="${1#*=}";  shift ;;
        --results_dir)   RESULTS_DIR="$2"; shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";  shift 2 ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$OUTPUT_DIR"

# ── Delegate to inline Python ─────────────────────────────────────────────────
# Bash cannot parse JSON natively — use a single self-contained Python block
# that reads all results, prints the table, and writes all output files.

python3 - "$RESULTS_DIR" "$OUTPUT_DIR" << 'PYEOF'
import json, sys
from pathlib import Path

results_dir = Path(sys.argv[1])
output_dir  = Path(sys.argv[2])

TASKS   = ["boolq", "hellaswag", "arc_challenge"]
METHODS = ["frozen", "pure_paft", "hybrid_paft", "lora_r8", "polar_r8",
           "lora_r64", "bitfit"]

TASK_LABELS = {
    "boolq":         "BoolQ",
    "hellaswag":     "HellaSwag",
    "arc_challenge": "ARC-C",
}
METHOD_LABELS = {
    "frozen":      "Frozen",
    "pure_paft":   "pure-PAFT (Ours)",
    "hybrid_paft": "hybrid-PAFT (Ours)",
    "lora_r8":     r"LoRA $r{=}8$",
    "polar_r8":    r"PoLAR $r{=}8$",
    "lora_r64":    r"LoRA $r{=}64$",
    "bitfit":      "BitFit",
}
PAFT_METHODS = {"pure_paft", "hybrid_paft"}
BASELINES    = {"boolq": 0.50, "hellaswag": 0.25, "arc_challenge": 0.25}

# ── Load results ──────────────────────────────────────────────────────────────

def load_acc(task, method):
    d = results_dir / task / method
    # 1. final/metrics.json
    for key in ("final_accuracy", "accuracy"):
        p = d / "final" / "metrics.json"
        if p.exists():
            v = json.loads(p.read_text()).get(key)
            if v is not None:
                return float(v), "final"
    # 2. epoch_0 fallback
    p = d / "epoch_0" / "metrics.json"
    if p.exists():
        v = json.loads(p.read_text()).get("accuracy")
        if v is not None:
            print(f"  WARNING: {task}/{method} — using epoch_0 (no final/)",
                  file=sys.stderr)
            return float(v), "epoch_0"
    return None, None

results = {}
for task in TASKS:
    results[task] = {}
    for method in METHODS:
        acc, source = load_acc(task, method)
        results[task][method] = {"accuracy": acc, "source": source} if acc else None

# ── Best per column ───────────────────────────────────────────────────────────

best_overall = {}
best_paft    = {}
for task in TASKS:
    bv, bm, bpv, bpm = -1, "", -1, ""
    for method in METHODS:
        e = results[task].get(method)
        if not e: continue
        a = e["accuracy"]
        if a > bv:  bv, bm = a, method
        if method in PAFT_METHODS and a > bpv: bpv, bpm = a, method
    best_overall[task] = (bm, bv)
    best_paft[task]    = (bpm, bpv)

# ── Text table ────────────────────────────────────────────────────────────────

mw, cw = 24, 13
header = f"{'Method':<{mw}}" + "".join(f"  {TASK_LABELS[t]:>{cw}}" for t in TASKS)
header += f"  {'Mean':>{cw}}"
sep = "─" * len(header)

lines = [sep, header, sep]
for method in METHODS:
    vals = [results[t].get(method, None) for t in TASKS]
    accs = [v["accuracy"] if v else None for v in vals]
    valid = [a for a in accs if a is not None]
    mean  = sum(valid)/len(valid) if valid else None
    row   = f"{METHOD_LABELS.get(method, method):<{mw}}"
    for a in accs:
        row += f"  {a:>{cw}.4f}" if a is not None else f"  {'—':>{cw}}"
    row += f"  {mean:>{cw}.4f}" if mean is not None else f"  {'—':>{cw}}"
    lines.append(row)
lines.append(sep)
lines.append("")
lines.append("Sanity checks (values must exceed random baseline):")
ok = True
for task in TASKS:
    base = BASELINES[task]
    for method in METHODS:
        e = results[task].get(method)
        if e and e["accuracy"] <= base + 0.02:
            lines.append(
                f"  WARNING: {method}/{task} = {e['accuracy']:.4f} "
                f"(<= baseline {base:.2f} + 0.02)"
            )
            ok = False
if ok:
    lines.append("  All values above random baseline. ✓")

text_table = "\n".join(lines)
print()
print("=" * 80)
print("  LLAMA-3.2-3B COMMONSENSE REASONING RESULTS")
print("=" * 80)
print(text_table)
print()

# Write text summary
txt_path = output_dir / "llama_results_summary.txt"
txt_path.write_text(text_table)
print(f"Text summary  →  {txt_path}")

# ── LaTeX table ───────────────────────────────────────────────────────────────

def fmt_cell(task, method, acc):
    s = f"{acc:.4f}"
    is_best      = best_overall[task][0] == method
    is_best_paft = best_paft[task][0]    == method
    if is_best and is_best_paft: return rf"\textbf{{\underline{{{s}}}}}"
    if is_best:                  return rf"\textbf{{{s}}}"
    if is_best_paft:             return rf"\underline{{{s}}}"
    return s

tex_lines = []
task_header = " & ".join(rf"\textbf{{{TASK_LABELS[t]}}}" for t in TASKS)
tex_lines.append(rf"\textbf{{Method}} & {task_header} & \textbf{{Mean}} \\")
tex_lines.append(r"\midrule")

prev_group = None
for method in METHODS:
    group = ("paft" if method in PAFT_METHODS else
             "frozen" if method == "frozen" else "additive")
    if prev_group is not None and group != prev_group:
        tex_lines.append(r"\midrule")
    prev_group = group

    cells    = []
    row_accs = []
    for task in TASKS:
        e = results[task].get(method)
        if e:
            row_accs.append(e["accuracy"])
            cells.append(fmt_cell(task, method, e["accuracy"]))
        else:
            row_accs.append(None)
            cells.append("--")

    valid    = [a for a in row_accs if a is not None]
    mean_str = f"{sum(valid)/len(valid):.4f}" if valid else "--"
    label    = METHOD_LABELS.get(method, method)
    tex_lines.append(" & ".join([label] + cells + [mean_str]) + r" \\")

latex = "\n".join(tex_lines)
tex_path = output_dir / "llama_results_table.tex"
tex_path.write_text(latex)
print(f"LaTeX table   →  {tex_path}")

# ── JSON summary ──────────────────────────────────────────────────────────────

summary = {
    "model":    "meta-llama/Llama-3.2-3B",
    "protocol": "log-likelihood multiple-choice (PoLAR protocol)",
    "epochs":   "2 (BoolQ, ARC-C); 1 (HellaSwag, compute-constrained)",
    "tasks":    TASKS,
    "methods":  METHODS,
    "results":  {
        task: {
            method: results[task][method]
            for method in METHODS
            if results[task].get(method)
        }
        for task in TASKS
    },
}
json_path = output_dir / "llama_results.json"
json_path.write_text(json.dumps(summary, indent=2))
print(f"JSON summary  →  {json_path}")

# ── Missing runs report ───────────────────────────────────────────────────────

missing = [(t, m) for t in TASKS for m in METHODS if not results[t].get(m)]
total   = len(TASKS) * len(METHODS)
found   = total - len(missing)
print(f"\nRuns found: {found}/{total}")
if missing:
    print(f"Missing:    {len(missing)}")
    for t, m in missing:
        print(f"  MISSING: {t}/{m}")

PYEOF

echo ""
echo "Done. Output written to $OUTPUT_DIR"
PYEOF
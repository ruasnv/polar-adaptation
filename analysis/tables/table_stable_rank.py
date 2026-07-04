#!/usr/bin/env python3
r"""
analysis/table_stable_rank.py

sr(W_eff) init, final, and Δsr averaged across all GLUE tasks.
This is the geometric summary table — Table 2 in the paper.

Output: results/analysis/table_stable_rank.tex  (complete, \input-ready)
"""
import json
import sys
from pathlib import Path

import numpy as np


from analysis.plot_style import METHOD_LABELS

OUT_DIR = Path("results/analysis")

METHOD_ORDER = [
    "frozen",
    "bitfit", "svf",
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "lora_r8", "lora_r64", "polar_r8",
    "full_ft",
]

GROUPS = [
    ["frozen"],
    ["bitfit", "svf"],
    ["pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"],
    ["lora_r8", "lora_r64", "polar_r8"],
    ["full_ft"],
]


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/metrics_cache.json not found.")

    with open(cache_path) as f:
        glue = json.load(f)["glue"]

    # Aggregate per method across all tasks
    summary: dict[str, dict[str, list]] = {}
    for task, methods in glue.items():
        for method, metrics in methods.items():
            summary.setdefault(method, {"init": [], "final": [], "delta_sr": [], "sr_delta_W": []})
            init_sr  = metrics.get("sr_Weff_init")
            final_sr = metrics.get("sr_Weff_final")
            sr_dw    = metrics.get("sr_deltaW_V")
            if init_sr is not None and final_sr is not None:
                summary[method]["init"].append(init_sr)
                summary[method]["final"].append(final_sr)
                summary[method]["delta_sr"].append(final_sr - init_sr)
            if sr_dw is not None:
                summary[method]["sr_delta_W"].append(sr_dw)

    # ── Terminal preview ───────────────────────────────────────────────────────
    print(f"\n{'Method':<26}  {'sr_init':>8}  {'sr_final':>9}  {'Δsr':>7}  {'sr(ΔW)':>8}")
    print("─" * 65)
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        d = summary[method]
        i  = np.mean(d["init"])    if d["init"]    else None
        f  = np.mean(d["final"])   if d["final"]   else None
        ds = np.mean(d["delta_sr"]) if d["delta_sr"] else None
        dw = np.mean(d["sr_delta_W"]) if d["sr_delta_W"] else None

        def p(v, fmt=".3f"): return f"{v:{fmt}}" if v is not None else "N/A"
        print(f"{METHOD_LABELS.get(method, method):<26}  {p(i):>8}  {p(f):>9}  "
              f"{p(ds, '+.3f'):>7}  {p(dw):>8}")

    # ── LaTeX — complete booktabs table ───────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Geometric health of $W_\mathrm{eff}$ before and after fine-tuning, "
        r"averaged across all GLUE tasks. "
        r"$sr(\Delta W_V)$ measures update structural complexity. "
        r"Higher $sr(W_\mathrm{eff})$ indicates better-preserved geometry.}"
    )
    lines.append(r"\label{tab:stable_rank}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & "
        r"$sr(W_\mathrm{eff})_\mathrm{init}$ & "
        r"$sr(W_\mathrm{eff})_\mathrm{final}$ & "
        r"$\Delta sr$ & "
        r"$sr(\Delta W_V)$ \\"
    )
    lines.append(r"\midrule")

    for g_idx, group in enumerate(GROUPS):
        for method in group:
            if method not in summary:
                continue
            d  = summary[method]
            i  = np.mean(d["init"])      if d["init"]      else None
            f  = np.mean(d["final"])     if d["final"]      else None
            ds = np.mean(d["delta_sr"])  if d["delta_sr"]  else None
            dw = np.mean(d["sr_delta_W"]) if d["sr_delta_W"] else None

            def cell(v, fmt): return f"{v:{fmt}}" if v is not None else r"\text{---}"
            label = METHOD_LABELS.get(method, method)
            lines.append(
                f"{label} & {cell(i, '.2f')} & {cell(f, '.2f')} & "
                f"{cell(ds, '+.2f')} & {cell(dw, '.2f')} \\\\"
            )
        if g_idx < len(GROUPS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "table_stable_rank.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {OUT_DIR}/table_stable_rank.tex")


if __name__ == "__main__":
    main()
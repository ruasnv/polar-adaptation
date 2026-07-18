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

# Methods whose W_O is untouched by construction (no weight-update mechanism
# ever reaches it): report "(untouched)" here instead of a number, rather
# than a computed ~0 that looks the same as "we checked and it's ~0" for
# methods where W_O genuinely IS trained (SVF, PoLAR, all 4 PAFT variants,
# Full FT). This is a real distinction: LoRA on DeBERTa targets query/value
# projections only, never output.dense, so its W_O delta being ~0 in the
# data is a structural fact, not evidence of anything about the method.
TRIVIAL_O_METHODS = {"frozen", "bitfit", "lora_r8", "lora_r64"}


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
            summary.setdefault(method, {"init": [], "final": [], "delta_sr": [], "sr_delta_W": [],
                                         "init_o": [], "final_o": [], "delta_sr_o": []})
            init_sr  = metrics.get("sr_Weff_init")
            final_sr = metrics.get("sr_Weff_final")
            sr_dw    = metrics.get("sr_deltaW_V")
            if init_sr is not None and final_sr is not None:
                summary[method]["init"].append(init_sr)
                summary[method]["final"].append(final_sr)
                summary[method]["delta_sr"].append(final_sr - init_sr)
            if sr_dw is not None:
                summary[method]["sr_delta_W"].append(sr_dw)

            init_sr_o  = metrics.get("sr_Weff_O_init")
            final_sr_o = metrics.get("sr_Weff_O_final")
            if init_sr_o is not None and final_sr_o is not None:
                summary[method]["init_o"].append(init_sr_o)
                summary[method]["final_o"].append(final_sr_o)
                summary[method]["delta_sr_o"].append(final_sr_o - init_sr_o)

    # ── Terminal preview ───────────────────────────────────────────────────────
    print(f"\n{'Method':<26}  {'sr_init':>8}  {'sr_final':>9}  {'Δsr':>7}  {'sr(ΔW)':>8}  {'Δsr_O':>12}")
    print("─" * 80)
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        d = summary[method]
        i  = np.mean(d["init"])    if d["init"]    else None
        f  = np.mean(d["final"])   if d["final"]   else None
        ds = np.mean(d["delta_sr"]) if d["delta_sr"] else None
        dw = np.mean(d["sr_delta_W"]) if d["sr_delta_W"] else None
        ds_o = np.mean(d["delta_sr_o"]) if d["delta_sr_o"] else None

        def p(v, fmt=".3f"): return f"{v:{fmt}}" if v is not None else "N/A"
        if method in TRIVIAL_O_METHODS:
            ds_o_str = "(untouched)"
        else:
            ds_o_str = p(ds_o, "+.3f")
        print(f"{METHOD_LABELS.get(method, method):<26}  {p(i):>8}  {p(f):>9}  "
              f"{p(ds, '+.3f'):>7}  {p(dw):>8}  {ds_o_str:>12}")

    # ── LaTeX — complete booktabs table ───────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Geometric health of $W_\mathrm{eff}$ before and after "
        r"fine-tuning, averaged across all GLUE tasks. "
        r"$sr(\Delta W_V)$ measures update structural complexity; higher "
        r"$sr(W_\mathrm{eff})$ indicates better-preserved geometry. "
        r"$\Delta sr_O$ reports the same quantity for $W_O$; "
        r"``(untouched)'' marks methods whose update mechanism never "
        r"reaches $W_O$, for which a $\sim\!0$ delta is a structural fact "
        r"rather than a measured result.}"
    )
    lines.append(r"\label{tab:stable_rank}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & "
        r"$sr(W_\mathrm{eff})_\mathrm{init}$ & "
        r"$sr(W_\mathrm{eff})_\mathrm{final}$ & "
        r"$\Delta sr$ & "
        r"$sr(\Delta W_V)$ & "
        r"$\Delta sr_O$ \\"
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
            ds_o = np.mean(d["delta_sr_o"]) if d["delta_sr_o"] else None

            def cell(v, fmt): return f"{v:{fmt}}" if v is not None else r"\text{---}"
            label = METHOD_LABELS.get(method, method)
            if method in TRIVIAL_O_METHODS:
                ds_o_cell = r"\text{--- (untouched)}"
            else:
                ds_o_cell = cell(ds_o, "+.2f")
            lines.append(
                f"{label} & {cell(i, '.2f')} & {cell(f, '.2f')} & "
                f"{cell(ds, '+.2f')} & {cell(dw, '.2f')} & {ds_o_cell} \\\\"
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
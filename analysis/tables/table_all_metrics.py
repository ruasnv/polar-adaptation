#!/usr/bin/env python3
r"""
analysis/table_all_metrics.py

Comprehensive multi-metric geometric profile, averaged across GLUE tasks.
This is the supplementary table (appendix).

Metrics:
  Δsr (%)      — percentage change in sr(W_eff) vs pretrained
  ΔEntropy     — change in spectral entropy vs frozen baseline
  ΔEffRank (%) — percentage change in effective rank vs frozen baseline
  CondNum      — final condition number σ_max/σ_min
  Isotropy     — final isotropy σ_min/σ_max

  participation_ratio and nuclear_norm_ratio are defined in stable_rank.py
  but excluded from the main results (see Section 2.4).

Output: results/analysis/table_all_metrics.tex  (complete, \input-ready)
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


def avg(lst):
    return float(np.mean(lst)) if lst else None


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        sys.exit("Error: results/analysis/metrics_cache.json not found.")

    with open(cache_path) as f:
        glue = json.load(f)["glue"]

    tasks = list(glue.keys())

    # ── Pretrained baselines from frozen ──────────────────────────────────────
    baseline_entropy:  dict[str, float] = {}
    baseline_eff_rank: dict[str, float] = {}
    for task in tasks:
        frozen = glue[task].get("frozen", {})
        e = frozen.get("spectral_entropy_Weff_final")
        r = frozen.get("effective_rank_Weff_final")
        if e is not None: baseline_entropy[task]  = e
        if r is not None: baseline_eff_rank[task] = r

    # ── Aggregate per method ──────────────────────────────────────────────────
    results: dict[str, dict[str, list]] = {}
    for method in METHOD_ORDER:
        results[method] = {
            "sr_delta_pct": [], "ent_delta": [], "er_delta_pct": [],
            "cond": [], "iso": [],
        }

    for task in tasks:
        for method in METHOD_ORDER:
            if method not in glue[task]:
                continue
            m = glue[task][method]
            r = results[method]

            init_sr  = m.get("sr_Weff_init")
            final_sr = m.get("sr_Weff_final")
            if init_sr and final_sr and init_sr > 0:
                r["sr_delta_pct"].append((final_sr - init_sr) / init_sr * 100)

            ent_f = m.get("spectral_entropy_Weff_final")
            ent_b = baseline_entropy.get(task)
            if ent_f is not None and ent_b is not None:
                r["ent_delta"].append(ent_f - ent_b)

            er_f = m.get("effective_rank_Weff_final")
            er_b = baseline_eff_rank.get(task)
            if er_f is not None and er_b is not None and er_b > 0:
                r["er_delta_pct"].append((er_f - er_b) / er_b * 100)

            cond = m.get("condition_number_final")
            if cond is not None and cond > 0:
                r["cond"].append(cond)
                r["iso"].append(1.0 / cond)

    # ── Terminal preview ───────────────────────────────────────────────────────
    print(f"\n{'Method':<24}  {'Δsr%':>7}  {'ΔEnt':>7}  {'ΔER%':>7}  "
          f"{'CondNum':>9}  {'Isotropy':>9}")
    print("─" * 78)
    for method in METHOD_ORDER:
        r = results.get(method, {})
        if not r.get("sr_delta_pct"): continue

        def p(key, fmt): v = avg(r.get(key,[])); return f"{v:{fmt}}" if v is not None else "N/A"
        print(f"{METHOD_LABELS.get(method, method):<24}  "
              f"{p('sr_delta_pct','+7.2f')}  {p('ent_delta','+7.3f')}  "
              f"{p('er_delta_pct','+7.2f')}  {p('cond','9.1e')}  "
              f"{p('iso','9.4f')}")

    # ── LaTeX — complete booktabs table ───────────────────────────────────────
    col_spec = "lrrrrr"

    header_cells = [
        r"\textbf{Method}",
        r"$\Delta sr$ (\%)",
        r"$\Delta H$",
        r"$\Delta\mathrm{ER}$ (\%)",
        r"$\kappa$",
        r"Isotropy",
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Multi-metric geometric profile averaged across GLUE tasks. "
        r"$\Delta sr$: percentage change in $sr(W_\mathrm{eff})$. "
        r"$\Delta H$, $\Delta\mathrm{ER}$: change in spectral entropy and effective rank "
        r"relative to the frozen (pretrained) baseline. "
        r"$\kappa = \sigma_\mathrm{max}/\sigma_\mathrm{min}$. "
        r"Isotropy $= 1/\kappa$.}"
    )
    lines.append(r"\label{tab:all_metrics}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for g_idx, group in enumerate(GROUPS):
        for method in group:
            r = results.get(method, {})
            if not r.get("sr_delta_pct"): continue

            def cell(key, fmt):
                v = avg(r.get(key, []))
                return f"{v:{fmt}}" if v is not None else r"\text{---}"

            label = METHOD_LABELS.get(method, method)
            cells = [
                label,
                cell("sr_delta_pct", "+.1f") + r"\%",
                cell("ent_delta", "+.3f"),
                cell("er_delta_pct", "+.1f") + r"\%",
                cell("cond", ".2e"),
                cell("iso", ".4f"),
            ]
            lines.append(" & ".join(cells) + r" \\")

        if g_idx < len(GROUPS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "table_all_metrics.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {OUT_DIR}/table_all_metrics.tex")


if __name__ == "__main__":
    main()
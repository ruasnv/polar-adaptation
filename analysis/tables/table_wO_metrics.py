#!/usr/bin/env python3
r"""
analysis/table_wO_metrics.py

Supplementary appendix table: W_O geometric profile, mirroring the five
metrics in table_all_metrics.py (Δsr%, ΔEntropy, ΔEffRank%, κ, Isotropy),
but for the output projection W_O instead of W_V.

Scope, by design — read this before adding methods:
  W_V remains the paper's primary, uniformly-reported metric throughout
  Results, since every method has SOME relationship to it (real change or
  genuine untouched baseline). W_O is only informative for methods that
  actually target the full OV circuit. This table is therefore restricted
  to METHODS_WITH_REAL_O — methods where a computed W_O delta reflects a
  real measurement, not a structural zero (Frozen, BitFit, and LoRA on
  DeBERTa never touch W_O; they are correctly excluded here, not just
  hidden). If you're tempted to add a method to METHODS_WITH_REAL_O,
  confirm first that its training actually updates W_O — don't add it just
  because build_cache.py happens to report a non-null value; a near-zero
  real measurement and a structural zero look identical in the data.

Output: results/analysis/table_wO_metrics.tex  (complete, \input-ready)
"""
import json
import sys
from pathlib import Path

import numpy as np

from analysis.plot_style import METHOD_LABELS

OUT_DIR = Path("results/analysis")

# Only methods whose training mechanism actually reaches W_O. Frozen, BitFit,
# and LoRA (DeBERTa target_modules = query/value only) are deliberately
# excluded — see table_stable_rank.py's TRIVIAL_O_METHODS for the flagship
# table's version of this same distinction.
METHODS_WITH_REAL_O = [
    "svf",
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "polar_r8",
    "full_ft",
]

GROUPS = [
    ["svf"],
    ["pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"],
    ["polar_r8"],
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

    # ── Pretrained baselines from frozen's W_O (if present) ───────────────────
    # NOTE: frozen is excluded from METHODS_WITH_REAL_O (it's the trivial
    # case by definition), but it's still the correct baseline source for
    # computing deltas, exactly as table_all_metrics.py uses frozen's W_V
    # as the baseline for every other method's ΔEntropy/ΔEffRank.
    baseline_entropy_o:  dict[str, float] = {}
    baseline_eff_rank_o: dict[str, float] = {}
    for task in tasks:
        frozen = glue[task].get("frozen", {})
        e = frozen.get("spectral_entropy_Weff_O_final")
        r = frozen.get("effective_rank_Weff_O_final")
        if e is not None: baseline_entropy_o[task]  = e
        if r is not None: baseline_eff_rank_o[task] = r

    if not baseline_entropy_o:
        print("Warning: no frozen spectral_entropy_Weff_O_final found in any task — "
              "ΔEntropy_O and ΔEffRank_O% will be null throughout. This is expected "
              "if frozen's geometric_health.pt predates the W_O extraction added to "
              "build_cache.py — rerun build_cache.py against current checkpoints.")

    # ── Aggregate per method ──────────────────────────────────────────────────
    results: dict[str, dict[str, list]] = {}
    for method in METHODS_WITH_REAL_O:
        results[method] = {
            "sr_delta_pct": [], "ent_delta": [], "er_delta_pct": [],
            "cond": [], "iso": [],
        }

    for task in tasks:
        for method in METHODS_WITH_REAL_O:
            if method not in glue[task]:
                continue
            m = glue[task][method]
            r = results[method]

            init_sr  = m.get("sr_Weff_O_init")
            final_sr = m.get("sr_Weff_O_final")
            if init_sr and final_sr and init_sr > 0:
                r["sr_delta_pct"].append((final_sr - init_sr) / init_sr * 100)

            ent_f = m.get("spectral_entropy_Weff_O_final")
            ent_b = baseline_entropy_o.get(task)
            if ent_f is not None and ent_b is not None:
                r["ent_delta"].append(ent_f - ent_b)

            er_f = m.get("effective_rank_Weff_O_final")
            er_b = baseline_eff_rank_o.get(task)
            if er_f is not None and er_b is not None and er_b > 0:
                r["er_delta_pct"].append((er_f - er_b) / er_b * 100)

            cond = m.get("condition_number_O_final")
            if cond is not None and cond > 0:
                r["cond"].append(cond)
                r["iso"].append(1.0 / cond)

    # ── Terminal preview ───────────────────────────────────────────────────────
    print(f"\n{'Method':<24}  {'Δsr_O%':>7}  {'ΔEnt_O':>7}  {'ΔER_O%':>7}  "
          f"{'CondNum_O':>10}  {'Isotropy_O':>10}")
    print("─" * 82)
    any_row = False
    for method in METHODS_WITH_REAL_O:
        r = results.get(method, {})
        if not r.get("sr_delta_pct"):
            continue
        any_row = True

        def p(key, fmt): v = avg(r.get(key,[])); return f"{v:{fmt}}" if v is not None else "N/A"
        print(f"{METHOD_LABELS.get(method, method):<24}  "
              f"{p('sr_delta_pct','+7.2f')}  {p('ent_delta','+7.3f')}  "
              f"{p('er_delta_pct','+7.2f')}  {p('cond','10.1e')}  "
              f"{p('iso','10.4f')}")
    if not any_row:
        print("  (no data — rerun build_cache.py against checkpoints that include W_O)")

    # ── LaTeX — complete booktabs table ───────────────────────────────────────
    col_spec = "lrrrrr"

    header_cells = [
        r"\textbf{Method}",
        r"$\Delta sr_O$ (\%)",
        r"$\Delta H_O$",
        r"$\Delta\mathrm{ER}_O$ (\%)",
        r"$\kappa_O$",
        r"Isotropy$_O$",
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Multi-metric geometric profile of $W_O$ (output "
        r"projection), averaged across GLUE tasks, restricted to methods "
        r"whose training mechanism actually reaches $W_O$; Frozen, BitFit, "
        r"and LoRA are omitted since their $W_O$ delta would be a "
        r"structural zero rather than a measurement. "
        r"$\Delta sr_O$: percentage change in $sr(W_{O,\mathrm{eff}})$. "
        r"$\Delta H_O$, $\Delta\mathrm{ER}_O$: change in spectral entropy "
        r"and effective rank of $W_O$ relative to the frozen baseline. "
        r"$\kappa_O = \sigma_\mathrm{max}/\sigma_\mathrm{min}$ of $W_O$; "
        r"Isotropy$_O$ is a separate per-head average of "
        r"$\sigma_\mathrm{min}/\sigma_\mathrm{max}$, not the reciprocal "
        r"of $\kappa_O$.}"
    )
    lines.append(r"\label{tab:wO_metrics}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for g_idx, group in enumerate(GROUPS):
        for method in group:
            r = results.get(method, {})
            if not r.get("sr_delta_pct"):
                continue

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
    with open(OUT_DIR / "table_wO_metrics.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {OUT_DIR}/table_wO_metrics.tex")


if __name__ == "__main__":
    main()
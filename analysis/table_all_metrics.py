#!/usr/bin/env python3
"""
analysis/table_all_metrics.py

Computes a comprehensive structural evaluation matrix across geometric metrics,
averaged across GLUE tasks.

Metrics reported:
  Δsr (%)       — percentage change in stable rank of W_eff
  ΔEntropy      — absolute change in spectral entropy (baseline from frozen method)
  ΔEffRank (%)  — percentage change in effective rank (baseline from frozen method)
  CondNum       — final absolute condition number σ_max/σ_min
  Isotropy      — final absolute isotropy σ_min/σ_max = 1/CondNum
  PartRatio     — from cache if available (requires build_cache.py update), else N/A
  NucNormRatio  — from cache if available (requires build_cache.py update), else N/A
"""
import json
from pathlib import Path
import numpy as np

METHOD_ORDER = [
    "frozen", "bitfit", "svf",
    "pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft",
    "lora_r8", "lora_r64", "polar_r8", "full_ft",
]


def main():
    cache_path = Path("results/analysis/metrics_cache.json")
    if not cache_path.exists():
        print("Error: Compile metrics_cache.json before running this script.")
        return

    with open(cache_path) as f:
        data = json.load(f)["glue"]

    tasks = list(data.keys())

    # ── Derive pretrained baselines from the frozen method ─────────────────
    # frozen does not adapt weight matrices, so its final geometric values
    # equal the pretrained initialisation values. This avoids hardcoded
    # approximations for metrics not stored at init time in the cache.
    baseline_entropy  = {}   # task → pretrained spectral entropy
    baseline_eff_rank = {}   # task → pretrained effective rank

    for task in tasks:
        frozen = data[task].get("frozen", {})
        if frozen.get("spectral_entropy_Weff_final") is not None:
            baseline_entropy[task]  = frozen["spectral_entropy_Weff_final"]
        if frozen.get("effective_rank_Weff_final") is not None:
            baseline_eff_rank[task] = frozen["effective_rank_Weff_final"]

    # ── Print table ────────────────────────────────────────────────────────
    print("\nSupplementary Table: Comprehensive Multi-Metric Geometric Profile")
    print("─" * 119)
    print(
        f"{'FT Method':<22} | {'Δsr (%)':>8} | {'ΔEntropy':>9} | {'ΔEffRank%':>10} | "
        f"{'CondNum':>9} | {'Isotropy':>9} | {'PartRatio':>10} | {'NNRatio':>9}"
    )
    print("─" * 119)

    latex_lines = []
    methods_in_data = set(m for t in data.values() for m in t.keys())

    for method in METHOD_ORDER:
        if method not in methods_in_data:
            continue

        sr_deltas, ent_deltas, er_deltas = [], [], []
        cond_vals, iso_vals               = [], []
        part_vals, nuc_vals               = [], []

        for task in tasks:
            if method not in data[task]:
                continue
            m = data[task][method]

            init_sr  = m.get("sr_Weff_init")
            final_sr = m.get("sr_Weff_final")
            cond     = m.get("condition_number_final")

            # ── Δsr (%) ───────────────────────────────────────────────────
            if init_sr and final_sr and init_sr > 0:
                sr_deltas.append(((final_sr - init_sr) / init_sr) * 100)

            # ── ΔEntropy — baseline from frozen, not a hardcoded constant ─
            ent_final = m.get("spectral_entropy_Weff_final")
            ent_base  = baseline_entropy.get(task)
            if ent_final is not None and ent_base is not None:
                ent_deltas.append(ent_final - ent_base)

            # ── ΔEffRank (%) — baseline from frozen, not hardcoded 34.0 ──
            er_final = m.get("effective_rank_Weff_final")
            er_base  = baseline_eff_rank.get(task)
            if er_final is not None and er_base is not None and er_base > 0:
                er_deltas.append(((er_final - er_base) / er_base) * 100)

            # ── CondNum and Isotropy (absolute final values) ───────────────
            if cond is not None and cond > 0:
                cond_vals.append(cond)
                iso_vals.append(1.0 / cond)   # iso = σ_min/σ_max = 1/κ

            # ── PartRatio — read from cache; N/A if not yet computed ───────
            pr = m.get("participation_ratio_final")
            if pr is not None:
                part_vals.append(pr)

            # ── NucNormRatio — read from cache; N/A if not yet computed ────
            nn = m.get("nuclear_norm_ratio")
            if nn is not None:
                nuc_vals.append(nn)

        if not sr_deltas:
            continue

        def avg(lst): return np.mean(lst) if lst else None
        def fmt_pct(v):  return f"{v:>+8.2f}" if v is not None else f"{'N/A':>8}"
        def fmt_abs(v):  return f"{v:>9.4f}" if v is not None else f"{'N/A':>9}"
        def fmt_sci(v):  return f"{v:>9.1e}" if v is not None else f"{'N/A':>9}"

        m_sr   = avg(sr_deltas)
        m_ent  = avg(ent_deltas)
        m_er   = avg(er_deltas)
        m_cond = avg(cond_vals)
        m_iso  = avg(iso_vals)
        m_part = avg(part_vals)
        m_nuc  = avg(nuc_vals)

        row = (
            f"{method:<22} | {fmt_pct(m_sr)} | {fmt_abs(m_ent)} | "
            f"{fmt_pct(m_er):>10} | {fmt_sci(m_cond)} | {fmt_abs(m_iso)} | "
            f"{fmt_abs(m_part):>10} | {fmt_abs(m_nuc):>9}"
        )
        print(row)

        def tex(v, fmt):
            return fmt % v if v is not None else r"\text{N/A}"
        tex_m = method.replace("_", r"\_")
        latex_lines.append(
            f"{tex_m} & {tex(m_sr, '%+.2f')}\\% & {tex(m_ent, '%+.3f')} & "
            f"{tex(m_er, '%+.2f')}\\% & {tex(m_cond, '%.1e')} & "
            f"{tex(m_iso, '%.4f')} & {tex(m_part, '%.3f')} & "
            f"{tex(m_nuc, '%.3f')} \\\\"
        )

    print("─" * 119)

    # ── Warn about N/A columns ─────────────────────────────────────────────
    sample_method = next(iter(methods_in_data))
    sample_task   = tasks[0]
    has_pr  = data[sample_task].get(sample_method, {}).get("participation_ratio_final") is not None
    has_nn  = data[sample_task].get(sample_method, {}).get("nuclear_norm_ratio") is not None
    if not has_pr:
        print("  NOTE: PartRatio shows N/A — add participation_ratio_final to build_cache.py "
              "and rebuild cache.")
    if not has_nn:
        print("  NOTE: NNRatio shows N/A — add nuclear_norm_ratio to build_cache.py "
              "and rebuild cache.")

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table_all_metrics.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print("LaTeX compiled to results/analysis/table_all_metrics.tex")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
analysis/build_paft_cache.py
FIXED: Expanded invariant manifold checking loops to cover both input (Q_V)
and output (Q_O) orthogonal projections systematically.
ADDED: S asymmetry measurement for hybrid-PAFT micro-rotation diagnosis.
"""
import json
import logging
from pathlib import Path
import torch
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


def main():
    from analysis.utils import setup_run_log
    setup_run_log("build_paft_cache")

    root = Path("results/glue")
    out_cache = Path("results/analysis/paft_cache.json")
    out_cache.parent.mkdir(parents=True, exist_ok=True)

    paft_cache = {}
    tasks = [d.name for d in root.iterdir() if d.is_dir()]

    for task in tasks:
        paft_cache[task] = {}
        task_dir = root / task
        paft_methods = [m.name for m in task_dir.iterdir() if m.is_dir() if "paft" in m.name]

        for method in paft_methods:
            method_dir = task_dir / method
            init_snap_p = method_dir / "init" / "paft_snapshot.pt"
            final_snap_p = method_dir / "final" / "paft_snapshot.pt"

            if not (init_snap_p.exists() and final_snap_p.exists()):
                continue

            try:
                init_snap  = torch.load(init_snap_p,  map_location="cpu")
                final_snap = torch.load(final_snap_p, map_location="cpu")

                missing = [k for k in ("Q_V", "Q_O")
                           if k not in init_snap or k not in final_snap]
                if missing:
                    log.warning(f"  Missing keys {missing} in snapshot for {task}/{method} — skipping")
                    continue

                num_layers = min(len(init_snap["Q_V"]), len(final_snap["Q_V"]), 12)

                # hybrid variants have S_V / S_O; pure variants have lam_V / lam_O
                is_hybrid = "hybrid" in method

                per_layer_info = []
                drifts_V       = []
                drifts_O       = []

                for layer_idx in range(num_layers):

                    # ── Value projection Q_V ──────────────────────────────────────────
                    f_qv   = final_snap["Q_V"][layer_idx].float()
                    i_qv   = init_snap["Q_V"][layer_idx].float()
                    diff_qv = (f_qv - i_qv).reshape(-1, f_qv.shape[-1])
                    drift_V = float(torch.norm(diff_qv, p="fro").item())
                    drifts_V.append(drift_V)

                    # ── Output projection Q_O ─────────────────────────────────────────
                    f_qo   = final_snap["Q_O"][layer_idx].float()
                    i_qo   = init_snap["Q_O"][layer_idx].float()
                    diff_qo = (f_qo - i_qo).reshape(-1, f_qo.shape[-1])
                    drift_O = float(torch.norm(diff_qo, p="fro").item())
                    drifts_O.append(drift_O)

                    # ── Eigenvalue shift magnitude (lam_V and lam_O) ──────────────────
                    # None = key doesn't exist (expected: hybrid variants don't use
                    # this parameterization). A caught exception is NOT the same as
                    # "no shift" — it must also stay None, and must be logged, or a
                    # real failure silently reads as "zero shift" in the tables.
                    delta_lam_V_mag = None
                    if "lam_V" in init_snap and "lam_V" in final_snap:
                        try:
                            delta_lam_V_mag = float(
                                torch.norm(
                                    final_snap["lam_V"][layer_idx].float()
                                    - init_snap["lam_V"][layer_idx].float(),
                                    p=2
                                ).item()
                            )
                        except Exception as ex:
                            log.error(f"  {task}/{method} layer {layer_idx}: "
                                      f"delta_lam_V computation failed ({ex}) — "
                                      f"writing null, not 0.0")

                    delta_lam_O_mag = None
                    if "lam_O" in init_snap and "lam_O" in final_snap:
                        try:
                            delta_lam_O_mag = float(
                                torch.norm(
                                    final_snap["lam_O"][layer_idx].float()
                                    - init_snap["lam_O"][layer_idx].float(),
                                    p=2
                                ).item()
                            )
                        except Exception as ex:
                            log.error(f"  {task}/{method} layer {layer_idx}: "
                                      f"delta_lam_O computation failed ({ex}) — "
                                      f"writing null, not 0.0")

                    # ── S asymmetry (hybrid-PAFT only) ────────────────────────────────
                    # Measures whether S_adapted drifted from the symmetric manifold.
                    # S is initialized as symmetric PSD (S₀). If training pushed it
                    # off the symmetric manifold, a micro-rotation Q' exists in the
                    # polar decomposition S_adapted = Q'S'.
                    #
                    # symmetry_ratio = ||S - Sᵀ||_F / ||S||_F  (per head, averaged)
                    #   < 0.01 → S stayed symmetric, no micro-rotation
                    #   > 0.10 → real asymmetry, micro-rotation Q' is non-trivial
                    #
                    # pure-PAFT uses diag(lam) which is always symmetric — skip.
                    #
                    # IMPORTANT: None (not 0.0) is the "computation didn't happen"
                    # value. 0.0 is this paper's own threshold for "perfectly
                    # symmetric" — the exact result the hypothesis wants — so a
                    # silently swallowed exception must never be able to produce it.
                    s_asym_V = None
                    s_asym_O = None

                    if is_hybrid and "S_V" in final_snap:
                        try:
                            S_V = final_snap["S_V"][layer_idx].float()  # [H, d, d]
                            asym_norms, frob_norms = [], []
                            for h in range(S_V.shape[0]):
                                S_h = S_V[h]
                                asym_norms.append((S_h - S_h.T).norm(p="fro").item())
                                frob_norms.append(S_h.norm(p="fro").item())
                            total_frob = sum(frob_norms)
                            s_asym_V = (sum(asym_norms) / total_frob) \
                                       if total_frob > 1e-12 else 0.0
                        except Exception as ex:
                            log.error(f"  {task}/{method} layer {layer_idx}: "
                                      f"S_V asymmetry computation failed ({ex}) — "
                                      f"writing null, not 0.0")

                    if is_hybrid and "S_O" in final_snap:
                        try:
                            S_O = final_snap["S_O"][layer_idx].float()  # [H, d, d]
                            asym_norms, frob_norms = [], []
                            for h in range(S_O.shape[0]):
                                S_h = S_O[h]
                                asym_norms.append((S_h - S_h.T).norm(p="fro").item())
                                frob_norms.append(S_h.norm(p="fro").item())
                            total_frob = sum(frob_norms)
                            s_asym_O = (sum(asym_norms) / total_frob) \
                                       if total_frob > 1e-12 else 0.0
                        except Exception as ex:
                            log.error(f"  {task}/{method} layer {layer_idx}: "
                                      f"S_O asymmetry computation failed ({ex}) — "
                                      f"writing null, not 0.0")

                    per_layer_info.append({
                        "layer":                  layer_idx,
                        "Q_V_drift":              drift_V,
                        "Q_O_drift":              drift_O,
                        "delta_lam_V_magnitude":  delta_lam_V_mag,
                        "delta_lam_O_magnitude":  delta_lam_O_mag,
                        "S_V_asymmetry_ratio":    s_asym_V,  # 0.0 for pure-PAFT
                        "S_O_asymmetry_ratio":    s_asym_O,
                    })

                def _mean_or_none(vals):
                    vals = [v for v in vals if v is not None]
                    return float(np.mean(vals)) if vals else None

                paft_cache[task][method] = {
                    # Q drift — should be 0.00e+00 for all PAFT variants
                    "Q_V_drift_mean":     float(np.mean(drifts_V)),
                    "Q_V_drift_max":      float(np.max(drifts_V)),
                    "Q_O_drift_mean":     float(np.mean(drifts_O)),
                    "Q_O_drift_max":      float(np.max(drifts_O)),
                    "Q_any_drift_max":    float(max(np.max(drifts_V),
                                                    np.max(drifts_O))),
                    # S asymmetry — non-zero only for hybrid variants if
                    # training pushed S off the symmetric manifold. null if
                    # the per-layer computation never succeeded for any layer
                    # (do not silently report this as symmetric).
                    "S_V_asymmetry_mean": _mean_or_none(
                        [r["S_V_asymmetry_ratio"] for r in per_layer_info]
                    ),
                    "S_O_asymmetry_mean": _mean_or_none(
                        [r["S_O_asymmetry_ratio"] for r in per_layer_info]
                    ),
                    "per_layer":          per_layer_info,
                }

            except Exception as ex:
                log.error(f"Error parsing polar snapshots for {task}/{method}: {ex}")

    with open(out_cache, "w") as f:
        json.dump(paft_cache, f, indent=2)
    log.info(f"[SUCCESS] PAFT cache compiled -> {out_cache}")


if __name__ == "__main__":
    main()
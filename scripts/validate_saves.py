"""
validate_saves.py — verify all required tensors are present after training.

Run this after any training sweep to catch missing tensors BEFORE attempting
analysis.  If a file is missing, the only fix is to re-run the experiment.

Usage:
    # Check all complete runs
    python scripts/validate_saves.py

    # Check a specific run
    python scripts/validate_saves.py --model gpt2_small --domain news --method hybrid_paft

    # Check all runs and report a summary
    python scripts/validate_saves.py --output_dir results/checkpoints

Exit code:
    0 — all checked runs are valid
    1 — one or more runs have missing files
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.schema import F


# ──────────────────────────────────────────────────────────────────────────────
# Required files per directory
# ──────────────────────────────────────────────────────────────────────────────

# Files required in init/ for ALL methods
INIT_ALL = [F.CONFIG, F.GEOMETRIC_HEALTH]

# Additional files required in init/ for surgery methods
INIT_PAFT = [F.DECOMP_INIT]
INIT_SVF  = [F.DECOMP_INIT]

# Files required in epoch_N/ for ALL methods
EPOCH_ALL = [F.METRICS, F.GEOMETRIC_HEALTH, F.MODEL, F.OPTIMIZER, F.SCHEDULER]

# Additional files required in epoch_N/ for PAFT methods
EPOCH_PAFT = [F.PAFT_SNAPSHOT]

# Files required in final/ for ALL methods
FINAL_ALL = [F.METRICS, F.GEOMETRIC_HEALTH, F.MODEL, F.ADAPTED_WEIGHTS, F.SENTINEL]

# Additional files required in final/ for PAFT methods
FINAL_PAFT = [F.PAFT_SNAPSHOT]

_PAFT_METHODS = {"pure_paft", "hybrid_paft", "safe_pure_paft", "safe_hybrid_paft"}
_SVF_METHODS  = {"svf"}


def _check_run(run_dir: Path, method: str) -> list[str]:
    """
    Check one completed run directory.
    Returns list of missing file paths (empty = all good).
    """
    missing = []
    is_paft = method in _PAFT_METHODS
    is_svf  = method in _SVF_METHODS

    # ── init/ ────────────────────────────────────────────────────────────────
    init_dir = run_dir / "init"
    for fname in INIT_ALL:
        if not (init_dir / fname).exists():
            missing.append(f"init/{fname}")

    if is_paft or is_svf:
        for fname in INIT_PAFT:
            if not (init_dir / fname).exists():
                missing.append(f"init/{fname}")

    # ── epoch_N/ ─────────────────────────────────────────────────────────────
    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    if not epoch_dirs:
        missing.append("epoch_*/ (no epoch directories found)")
    else:
        for epoch_dir in epoch_dirs:
            n = epoch_dir.name
            for fname in EPOCH_ALL:
                if not (epoch_dir / fname).exists():
                    missing.append(f"{n}/{fname}")
            if is_paft:
                for fname in EPOCH_PAFT:
                    if not (epoch_dir / fname).exists():
                        missing.append(f"{n}/{fname}")

    # ── final/ ───────────────────────────────────────────────────────────────
    final_dir = run_dir / "final"
    for fname in FINAL_ALL:
        if not (final_dir / fname).exists():
            missing.append(f"final/{fname}")
    if is_paft:
        for fname in FINAL_PAFT:
            if not (final_dir / fname).exists():
                missing.append(f"final/{fname}")

    return missing


def _discover_runs(output_dir: Path) -> list[tuple[str, str, str, Path]]:
    """
    Walk output_dir for all (model, domain, method, run_dir) tuples that
    have a training_complete sentinel.
    """
    runs = []
    for sentinel in sorted(output_dir.glob("*/*/*/final/training_complete")):
        run_dir = sentinel.parent.parent
        method  = run_dir.name
        domain  = run_dir.parent.name
        model   = run_dir.parent.parent.name
        runs.append((model, domain, method, run_dir))
    return runs


def parse_args():
    p = argparse.ArgumentParser(description="Validate PAFT checkpoint integrity.")
    p.add_argument("--output_dir", default="results/checkpoints",
                   help="Root directory to scan (default: results/checkpoints)")
    p.add_argument("--model",  default=None, help="Check one specific model")
    p.add_argument("--domain", default=None, help="Check one specific domain")
    p.add_argument("--method", default=None, help="Check one specific method")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ── Select runs to check ─────────────────────────────────────────────────
    if args.model and args.domain and args.method:
        run_dir = output_dir / args.model / args.domain / args.method
        if not run_dir.exists():
            print(f"ERROR: run directory not found: {run_dir}")
            return 1
        runs_to_check = [(args.model, args.domain, args.method, run_dir)]
    else:
        runs_to_check = _discover_runs(output_dir)
        # Filter if partial args given
        if args.model:
            runs_to_check = [r for r in runs_to_check if r[0] == args.model]
        if args.domain:
            runs_to_check = [r for r in runs_to_check if r[1] == args.domain]
        if args.method:
            runs_to_check = [r for r in runs_to_check if r[2] == args.method]

    if not runs_to_check:
        print(f"No complete runs found in: {output_dir}")
        return 0

    print(f"Validating {len(runs_to_check)} run(s) ...\n")

    # ── Check each run ────────────────────────────────────────────────────────
    all_ok     = True
    n_ok       = 0
    n_bad      = 0

    for model, domain, method, run_dir in runs_to_check:
        missing = _check_run(run_dir, method)
        label   = f"{model}/{domain}/{method}"

        if missing:
            all_ok = False
            n_bad += 1
            print(f"  FAIL  {label}")
            for f in missing:
                print(f"        MISSING: {f}")
        else:
            n_ok += 1
            print(f"  OK    {label}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Total checked: {len(runs_to_check)}")
    print(f"  OK:            {n_ok}")
    print(f"  FAILED:        {n_bad}")

    if all_ok:
        print("  All checkpoints valid — safe to run analysis.")
        return 0
    else:
        print("  Some files missing — re-run affected experiments before analysis.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
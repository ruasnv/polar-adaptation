"""
run_sweep.py — launch the full 100-run experiment matrix.

Iterates over every (model, domain, method) combination, skips runs that
already have a sentinel file, and calls run_experiment.py for each remaining
run.

Runs are executed sequentially — this is a single-GPU machine.  GPU is
released between runs via method.cleanup().

Usage:
    # Full sweep (100 runs — takes days)
    python scripts/run_sweep.py

    # Single model only
    python scripts/run_sweep.py --models gpt2_small

    # Specific methods only
    python scripts/run_sweep.py --methods hybrid_paft pure_paft

    # Dry run — print what would run without running it
    python scripts/run_sweep.py --dry_run

    # Resume — skip any run with a sentinel file (default behaviour)
    python scripts/run_sweep.py  # already resumes by default

Ablation runs are separate — call them explicitly:
    python scripts/run_sweep.py --ablation bias
    python scripts/run_sweep.py --ablation dial
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Experiment matrix ─────────────────────────────────────────────────────────

MODELS = ["gpt2_small", "gpt2_medium"]

DOMAINS = ["news", "legal", "biomedical", "code"]

METHODS = [
    "frozen",
    "full_finetune",
    "bitfit",
    "svf",
    "lora_r8",
    "lora_r64",
    "polar",
    "pure_paft",
    "hybrid_paft",
    "safe_pure_paft",
    "safe_hybrid_paft",
]

# Bias ablation: pure_paft base, news domain, gpt2_small only
BIAS_ABLATION_VARIANTS = [
    "bias_ablation_no_bias",
    "bias_ablation_attn_only",
    "bias_ablation_mlp_only",
    "bias_ablation_ln_only",
    "bias_ablation_all",
]

# Dial ablation: hybrid_paft with rotation penalty, 3 domains, gpt2_small
DIAL_ABLATION_PENALTIES = ["0.0", "0.01", "0.1", "1.0", "10.0"]
DIAL_ABLATION_DOMAINS   = ["news", "legal", "biomedical"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_complete(output_dir: Path, model: str, domain: str, method: str) -> bool:
    sentinel = output_dir / model / domain / method / "final" / "training_complete"
    return sentinel.exists()


def _run(model: str, domain: str, method: str, args: argparse.Namespace) -> int:
    """Run one experiment.  Returns subprocess exit code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
        "--model",  model,
        "--domain", domain,
        "--method", method,
        "--seed",   str(args.seed),
        "--output_dir", args.output_dir,
        "--log_dir",    args.log_dir,
        "--skip_if_complete",
    ]
    if args.max_steps:
        cmd += ["--max_steps", str(args.max_steps)]

    print(f"\n{'='*70}")
    print(f"  {model} / {domain} / {method}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  → {status}  ({elapsed/60:.1f} min)")
    return result.returncode


def parse_args():
    p = argparse.ArgumentParser(description="Run the full PAFT experiment sweep.")
    p.add_argument("--models",  nargs="+", default=MODELS,
                   choices=MODELS, help="Models to sweep (default: all)")
    p.add_argument("--domains", nargs="+", default=DOMAINS,
                   choices=DOMAINS, help="Domains to sweep (default: all)")
    p.add_argument("--methods", nargs="+", default=METHODS,
                   help="Methods to sweep (default: all 11)")
    p.add_argument("--ablation", choices=["bias", "dial"], default=None,
                   help="Run ablation sweep instead of main matrix")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", default="results/checkpoints")
    p.add_argument("--log_dir",    default="results/logs")
    p.add_argument("--max_steps",  type=int, default=None,
                   help="Smoke-test mode: limit each run to N steps")
    p.add_argument("--dry_run", action="store_true",
                   help="Print runs without executing them")
    return p.parse_args()


def main():
    args  = parse_args()
    odir  = Path(args.output_dir)

    # ── Build run list ────────────────────────────────────────────────────────
    runs = []

    if args.ablation == "bias":
        # Bias ablation: gpt2_small × news × 5 bias variants
        for method in BIAS_ABLATION_VARIANTS:
            runs.append(("gpt2_small", "news", method))

    elif args.ablation == "dial":
        # Dial ablation: gpt2_small × 3 domains × 5 penalty values
        for domain in DIAL_ABLATION_DOMAINS:
            for penalty in DIAL_ABLATION_PENALTIES:
                method = f"dial_ablation_{penalty.replace('.', '_')}"
                runs.append(("gpt2_small", domain, method))

    else:
        # Main matrix: all models × domains × methods
        for model in args.models:
            for domain in args.domains:
                for method in args.methods:
                    runs.append((model, domain, method))

    # ── Count / report ────────────────────────────────────────────────────────
    total     = len(runs)
    complete  = sum(1 for m, d, mth in runs if _is_complete(odir, m, d, mth))
    remaining = total - complete

    print(f"\nPAFT Sweep — {total} total runs")
    print(f"  Complete:  {complete}")
    print(f"  Remaining: {remaining}")

    if args.dry_run:
        print("\n[dry_run] Would run:")
        for model, domain, method in runs:
            done = " (done)" if _is_complete(odir, model, domain, method) else ""
            print(f"  {model} / {domain} / {method}{done}")
        return 0

    # ── Execute ───────────────────────────────────────────────────────────────
    failures = []
    t_sweep  = time.time()

    for i, (model, domain, method) in enumerate(runs, 1):
        if _is_complete(odir, model, domain, method):
            print(f"[{i}/{total}] skip (complete): {model}/{domain}/{method}")
            continue

        print(f"[{i}/{total}] running: {model}/{domain}/{method}")
        rc = _run(model, domain, method, args)
        if rc != 0:
            failures.append((model, domain, method))

    elapsed = time.time() - t_sweep
    print(f"\n{'='*70}")
    print(f"Sweep complete in {elapsed/3600:.1f}h")
    print(f"  Runs attempted: {remaining}")
    print(f"  Failures: {len(failures)}")
    if failures:
        print("  Failed runs:")
        for f in failures:
            print(f"    {f[0]}/{f[1]}/{f[2]}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
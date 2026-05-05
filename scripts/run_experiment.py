"""
run_experiment.py — entry point for a single training run.

Usage:
    python scripts/run_experiment.py \
        --model  gpt2_small \
        --domain news \
        --method hybrid_paft \
        [--seed 42] \
        [--output_dir results/checkpoints] \
        [--max_steps N]          # truncate for smoke-testing

One call = one cell in the experiment matrix.
run_sweep.py calls this for every (model, domain, method) triple.

This script is intentionally thin — no logic lives here.
All configuration comes from configs/, all logic lives in paft/.
"""

import argparse
import sys
import time
from pathlib import Path

# ── project root on sys.path so `import paft` works ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.utils.config       import get_config
from paft.utils.log_utils    import setup_logging
from paft.utils.reproducibility import set_seed
from paft.utils.device       import get_device, log_vram_usage, reset_peak_vram
from paft.methods            import get_method
from paft.training.trainer   import Trainer


# ── domain → data module ─────────────────────────────────────────────────────
def _get_data_module(domain: str, cfg: dict, hf_name: str):
    if domain == "news":
        from paft.data.news import NewsDataModule
        return NewsDataModule(cfg, hf_name)
    elif domain == "legal":
        from paft.data.legal import LegalDataModule
        return LegalDataModule(cfg, hf_name)
    elif domain == "biomedical":
        from paft.data.biomedical import BiomedicalDataModule
        return BiomedicalDataModule(cfg, hf_name)
    elif domain == "code":
        from paft.data.code import CodeDataModule
        return CodeDataModule(cfg, hf_name)
    else:
        raise ValueError(f"Unknown domain '{domain}'. Choose: news, legal, biomedical, code")


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run one PAFT experiment.")
    p.add_argument("--model",      required=True,
                   choices=["gpt2_small", "gpt2_medium"],
                   help="Model variant")
    p.add_argument("--domain",     required=True,
                   choices=["news", "legal", "biomedical", "code"],
                   help="Target domain")
    p.add_argument("--method",     required=True,
                   help="Fine-tuning method (must match configs/methods/*.yaml)")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--output_dir", default="results/checkpoints",
                   help="Root directory for checkpoints (default: results/checkpoints)")
    p.add_argument("--log_dir",    default="results/logs",
                   help="Directory for .log files (default: results/logs)")
    p.add_argument("--max_steps",  type=int, default=None,
                   help="Truncate training to N steps — for smoke-testing only")
    p.add_argument("--skip_if_complete", action="store_true",
                   help="Exit 0 silently if this run already has a sentinel file")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Experiment identity — used for directory names and log files
    experiment_id = f"{args.model}__{args.domain}__{args.method}"
    run_dir = Path(args.output_dir) / args.model / args.domain / args.method

    # Early exit if already complete
    if args.skip_if_complete and (run_dir / "final" / "training_complete").exists():
        print(f"[skip] {experiment_id} already complete.")
        return 0

    # Logging — must be first so all subsequent logger calls are captured
    setup_logging(
        experiment_name=experiment_id,
        log_dir=Path(args.log_dir) / args.model / args.domain,
    )

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"{'='*60}")
    logger.info(f"Experiment: {experiment_id}")
    logger.info(f"Run dir:    {run_dir}")
    logger.info(f"Seed:       {args.seed}")
    logger.info(f"{'='*60}")

    # Reproducibility
    set_seed(args.seed)

    # Config — merge base + model + domain + method YAMLs
    cfg = get_config(
        model  = args.model,
        domain = args.domain,
        method = args.method,
        project_root = PROJECT_ROOT,
    )

    # Inject CLI overrides that aren't in YAML
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
        logger.info(f"Smoke-test mode: max_steps={args.max_steps}")

    # HuggingFace model name
    hf_name = cfg["model"]["hf_name"]   # e.g. "gpt2" or "gpt2-medium"

    # Device
    device = get_device()
    reset_peak_vram()

    # ── Method ───────────────────────────────────────────────────────────────
    method = get_method(args.method, cfg)

    logger.info(f"Building method '{args.method}' from '{hf_name}' ...")
    t0 = time.time()
    method.build(hf_name, device)
    logger.info(f"Build complete in {time.time()-t0:.1f}s")
    log_vram_usage(f"{args.method}/post-build")

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading '{args.domain}' data ...")
    data = _get_data_module(args.domain, cfg, hf_name)
    train_loader = data.get_train_loader()
    val_loader   = data.get_val_loader()
    logger.info(
        f"Data ready — "
        f"train: {data.num_train_samples()} samples, "
        f"val: {data.num_val_samples()} samples"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        method       = method,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        run_dir      = run_dir,
    )

    logger.info("Starting training ...")
    t_train = time.time()
    final_metrics = trainer.train()
    elapsed = time.time() - t_train

    logger.info(f"Training complete in {elapsed/60:.1f} min")
    logger.info(f"Final metrics: {final_metrics}")
    log_vram_usage(f"{args.method}/final")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    method.cleanup()
    logger.info(f"VRAM released. Run complete: {experiment_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
fix_llama_epoch_fairness.py

For each LLaMA task, finds the minimum epoch count actually completed
across ALL methods on that task, then for any method whose final/
currently reflects a LATER epoch than that minimum, overwrites final/'s
analysis-relevant files with a copy of the minimum-epoch directory's
files — so every method on a given task is compared at the same,
fair epoch count.

Does NOT delete the extra epoch directories (e.g. epoch_2/ for BoolQ
PAFT methods) — they stay on disk for provenance/audit. Only final/'s
contents are overwritten.

SAFETY:
  - Dry-run by default. Nothing is written unless you pass --apply.
  - Backs up the current final/ to final_backup_pre_epoch_fix/ before
    overwriting anything, and skips the backup step (doesn't overwrite
    an existing backup) if run twice.
  - Only copies files the analysis pipeline actually reads:
    metrics.json, adapted_weights.pt, geometric_health.pt,
    paft_snapshot.pt (whichever exist). Does NOT touch config.json
    (will still describe the original epoch count — harmless metadata,
    not read by any analysis script) or training-resume artifacts
    (adapter.pt, optimizer.pt, scheduler.pt) or the training_complete
    sentinel.

Usage:
    python3 fix_llama_epoch_fairness.py             # dry run, report only
    python3 fix_llama_epoch_fairness.py --apply      # actually write changes
"""
import argparse
import json
import shutil
from pathlib import Path

TASKS = ["boolq", "hellaswag", "arc_challenge"]
METHODS = ["frozen", "pure_paft", "hybrid_paft", "lora_r8", "polar_r8",
           "lora_r64", "bitfit"]

ANALYSIS_FILES = ["metrics.json", "adapted_weights.pt", "geometric_health.pt",
                   "paft_snapshot.pt"]


def epoch_num(d: Path):
    parts = d.name.split("_")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1


def find_epoch_dirs(method_dir: Path):
    return sorted(method_dir.glob("epoch_*"), key=epoch_num)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/llama", type=Path)
    p.add_argument("--apply", action="store_true",
                    help="Actually write changes. Without this flag, only "
                         "reports what would change.")
    args = p.parse_args()

    for task in TASKS:
        task_dir = args.results_dir / task
        if not task_dir.exists():
            continue

        method_max_epoch = {}
        for method in METHODS:
            method_dir = task_dir / method
            if not method_dir.exists():
                continue
            epoch_dirs = find_epoch_dirs(method_dir)
            if not epoch_dirs:
                continue  # e.g. frozen — no epochs to compare
            method_max_epoch[method] = epoch_num(epoch_dirs[-1])

        if not method_max_epoch:
            continue

        min_epoch = min(method_max_epoch.values())
        print(f"\n{'='*70}\n{task}: per-method max epoch = {method_max_epoch}")
        print(f"  Fair common epoch for this task: epoch_{min_epoch}")

        for method, max_ep in method_max_epoch.items():
            if max_ep <= min_epoch:
                print(f"  {method}: already at epoch_{max_ep} == minimum — no change needed")
                continue

            method_dir = task_dir / method
            target_epoch_dir = method_dir / f"epoch_{min_epoch}"
            final_dir = method_dir / "final"
            backup_dir = method_dir / "final_backup_pre_epoch_fix"

            print(f"  {method}: final/ currently reflects epoch_{max_ep}, "
                  f"needs to be epoch_{min_epoch}")

            if not target_epoch_dir.exists():
                print(f"    ERROR: {target_epoch_dir} does not exist — cannot fix, skipping")
                continue

            old_metrics_p = final_dir / "metrics.json"
            new_metrics_p = target_epoch_dir / "metrics.json"
            old_acc = new_acc = None
            if old_metrics_p.exists():
                d = json.loads(old_metrics_p.read_text())
                old_acc = d.get("final_accuracy", d.get("accuracy"))
            if new_metrics_p.exists():
                d = json.loads(new_metrics_p.read_text())
                new_acc = d.get("accuracy")
            print(f"    accuracy: {old_acc} (epoch_{max_ep}, current final/) "
                  f"-> {new_acc} (epoch_{min_epoch}, corrected)")

            if not args.apply:
                print(f"    [DRY RUN] would back up final/ -> final_backup_pre_epoch_fix/, "
                      f"then copy epoch_{min_epoch}/{{{', '.join(ANALYSIS_FILES)}}} -> final/")
                continue

            # ── Actually apply ────────────────────────────────────────────
            if backup_dir.exists():
                print(f"    backup already exists at {backup_dir} — skipping "
                      f"backup step (already backed up from a previous run?)")
            else:
                shutil.copytree(final_dir, backup_dir)
                print(f"    backed up final/ -> {backup_dir}")

            for fname in ANALYSIS_FILES:
                src = target_epoch_dir / fname
                dst = final_dir / fname
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"    copied {fname} from epoch_{min_epoch} -> final/")
                else:
                    print(f"    ({fname} not found in epoch_{min_epoch}, skipped)")

            print(f"    DONE — {method}/final/ now reflects epoch_{min_epoch}. "
                  f"epoch_{max_ep}/ itself was left untouched on disk.")

    if not args.apply:
        print("\n\nThis was a DRY RUN — nothing was written. Review the changes "
              "above, then rerun with --apply to actually write them.")


if __name__ == "__main__":
    main()
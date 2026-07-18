#!/usr/bin/env python3
"""
patch_llama_metrics_metadata.py

fix_llama_epoch_fairness.py correctly copied epoch_N/metrics.json into
final/ for methods needing the epoch fix — but epoch-level metrics.json
only ever contains {"epoch", "train_loss", "accuracy"}, never the
"task"/"method"/"trainable_params" keys that the original top-level
final/metrics.json had. This restores those three metadata keys from
the backup, WITHOUT touching the corrected accuracy/epoch/train_loss
values that are already correct.

Usage:
    python3 patch_llama_metrics_metadata.py
"""
import json
from pathlib import Path

# (task, method) pairs that were actually fixed by fix_llama_epoch_fairness.py
FIXED = [("boolq", "pure_paft"), ("boolq", "hybrid_paft")]

RESULTS_DIR = Path("results/llama")


def main():
    for task, method in FIXED:
        method_dir = RESULTS_DIR / task / method
        final_p = method_dir / "final" / "metrics.json"
        backup_p = method_dir / "final_backup_pre_epoch_fix" / "metrics.json"

        if not final_p.exists() or not backup_p.exists():
            print(f"{task}/{method}: missing final/ or backup metrics.json, skipping")
            continue

        current = json.loads(final_p.read_text())
        backup = json.loads(backup_p.read_text())

        restored = 0
        for key in ("task", "method", "trainable_params"):
            if key not in current and key in backup:
                current[key] = backup[key]
                restored += 1

        # Keep it honest: this file's accuracy now reflects an EARLIER
        # epoch than the original run's true final epoch. Record that
        # explicitly rather than silently presenting it as an unmodified
        # final-epoch result.
        current["_epoch_fairness_note"] = (
            "metrics.json corrected to reflect the minimum common epoch "
            "across methods on this task (epoch fairness fix), not the "
            "original run's actual final epoch."
        )

        final_p.write_text(json.dumps(current, indent=2))
        print(f"{task}/{method}: restored {restored} metadata key(s), "
              f"wrote note. Final content: {current}")


if __name__ == "__main__":
    main()
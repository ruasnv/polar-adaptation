#!/usr/bin/env python3
"""
check_merge_freshness.py

Three-way comparison for the flagged epoch-1 LoRA entries:
  (a) what's currently in metrics_cache.json's per_epoch
  (b) what's actually stored on disk in epoch_1/geometric_health_merged.pt
  (c) what you get if you recompute the merge fresh, right now, from the
      adapter + base model

If (a) != (b): patch_metrics_cache.py was never applied (or applied to a
different cache file) — the cache has stale/wrong per-epoch data even
though the correct merged file exists on disk.

If (a) == (b) but both != (c): the merged .pt file itself is stale —
compute_lora_epoch_sr.py needs to be rerun with --force for this checkpoint.

If (a) == (b) == (c): everything is consistent and current — the near-
pretrained value is real, not a pipeline gap.

Usage:
    python3 check_merge_freshness.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel

sys.path.insert(0, "..")
from analysis.compute_lora_epoch_sr import (
    get_checkpoint_epoch_map, compute_merged_sr, METHOD_CONFIG, BASE_MODEL,
)

CACHE_PATH = Path("../results/analysis/metrics_cache.json")
FLAGGED = [("mrpc", "lora_r8", 1), ("rte", "lora_r8", 1)]


def main():
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    print("Loading base model (needed for fresh recomputation)...")
    base_model = AutoModel.from_pretrained(BASE_MODEL)
    base_state = base_model.state_dict()

    for task, method, epoch in FLAGGED:
        print(f"\n{'='*70}\n{task}/{method} epoch {epoch}\n{'='*70}")

        # (a) what's in the cache right now
        entry = cache.get("glue", {}).get(task, {}).get(method, {})
        cache_val = None
        for rec in entry.get("per_epoch", []):
            if rec.get("epoch") == epoch:
                cache_val = rec.get("sr_Weff")
        print(f"(a) metrics_cache.json per_epoch value: {cache_val}")

        # (b) what's on disk in the merged file
        method_dir = Path(f"results/glue/{task}/{method}")
        merged_path = method_dir / f"epoch_{epoch}" / "geometric_health_merged.pt"
        disk_val = None
        if merged_path.exists():
            data = torch.load(merged_path, map_location="cpu", weights_only=True)
            disk_val = data.get("mean_sr_V")
            print(f"(b) on-disk {merged_path.name}: {disk_val}")
        else:
            print(f"(b) {merged_path} does NOT exist on disk")

        # (c) freshly recomputed, right now, from the adapter
        epoch_map = get_checkpoint_epoch_map(method_dir)
        if epoch not in epoch_map:
            print(f"(c) no checkpoint found for epoch {epoch}, cannot recompute")
            continue
        cfg = METHOD_CONFIG[method]
        fresh = compute_merged_sr(base_state, epoch_map[epoch], cfg["rank"], cfg["alpha"])
        fresh_val = fresh["mean_sr_V"]
        print(f"(c) freshly recomputed right now: {fresh_val}")

        # Verdict
        def close(a, b, tol=0.01):
            return a is not None and b is not None and abs(a - b) < tol

        if not close(cache_val, disk_val):
            print("\nVERDICT: (a) != (b) — patch_metrics_cache.py was not "
                  "applied to this cache file. The correct merged value "
                  "exists on disk but never made it into metrics_cache.json. "
                  "Fix: rerun patch_metrics_cache.py against this exact cache.")
        elif not close(disk_val, fresh_val):
            print("\nVERDICT: (a)==(b) but both != (c) — the on-disk merged "
                  "file itself is stale. Fix: rerun compute_lora_epoch_sr.py "
                  "--force for this checkpoint, then patch_metrics_cache.py again.")
        else:
            print("\nVERDICT: (a)==(b)==(c) — fully consistent and current. "
                  "The near-pretrained value is real, not a pipeline gap.")


if __name__ == "__main__":
    main()
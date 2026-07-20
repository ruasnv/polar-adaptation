#!/usr/bin/env python3
"""
check_lora_b_zero.py

Directly answers: for the two flagged "LORA EPOCH UNCHANGED" entries
(mrpc/lora_r8 epoch 1, rte/lora_r8 epoch 1), is the epoch-1 adapter's
lora_B matrix actually trained (nonzero), or is it still literally the
zero-init default (meaning the merge genuinely fell back to the base
weight, and the flag is a real bug)?

Usage:
    python3 check_lora_b_zero.py
"""
import sys
from pathlib import Path

from safetensors.torch import load_file

sys.path.insert(0, "..")
from analysis.compute_lora_epoch_sr import get_checkpoint_epoch_map

FLAGGED = [
    ("mrpc", "lora_r8", 1),
    ("rte", "lora_r8", 1),
]

LAYERS_TO_CHECK = [0, 5, 11]


def main():
    for task, method, epoch in FLAGGED:
        method_dir = Path(f"results/glue/{task}/{method}")
        if not method_dir.exists():
            print(f"{task}/{method}: directory not found, skipping")
            continue

        epoch_map = get_checkpoint_epoch_map(method_dir)
        if epoch not in epoch_map:
            print(f"{task}/{method}: no checkpoint found for epoch {epoch}")
            continue

        adapter_path = epoch_map[epoch]
        print(f"\n{task}/{method} epoch {epoch}")
        print(f"  adapter: {adapter_path}")

        adapter = load_file(adapter_path)
        any_nonzero = False

        for layer_idx in LAYERS_TO_CHECK:
            key = (
                f"base_model.model.deberta.encoder.layer.{layer_idx}"
                f".attention.self.value_proj.lora_B.weight"
            )
            if key not in adapter:
                print(f"  layer {layer_idx}: key not found — check the key "
                      f"format matches your actual adapter file")
                continue
            max_abs = adapter[key].abs().max().item()
            print(f"  layer {layer_idx}  lora_B max|abs| = {max_abs:.6e}")
            if max_abs > 1e-6:
                any_nonzero = True

        if any_nonzero:
            print(f"  VERDICT: real, nonzero trained values — this is "
                  f"legitimate early-training data, not the base-weight "
                  f"fallback bug.")
        else:
            print(f"  VERDICT: all checked layers are ~zero — this IS the "
                  f"base-weight fallback bug, still live for this "
                  f"checkpoint. Needs investigation (was this checkpoint "
                  f"written before the compute_lora_epoch_sr.py fix, or "
                  f"does the fix have a remaining gap?).")


if __name__ == "__main__":
    main()
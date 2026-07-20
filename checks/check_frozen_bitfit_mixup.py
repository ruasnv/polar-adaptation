#!/usr/bin/env python3
"""
check_frozen_bitfit_mixup.py

geometric_heatmaps.pdf's dump data shows Frozen and BitFit with byte-
identical, nonzero, layer-varying per-layer values on STS-B — impossible
if these methods genuinely leave W_V untouched. This checks the raw files
directly, bypassing metrics_cache.json, to isolate whether this is:
  (a) a literal file-level mixup (same file duplicated/symlinked across
      method directories) — checked via file hash comparison
  (b) a real difference between init and final for a supposedly-frozen
      method — checked via direct per-layer stable-rank extraction from
      the raw geometric_health.pt files
  (c) something specific to how build_cache.py reads/caches these files
      (in which case (a) and (b) would both come back clean here, and
      the bug is downstream of the raw data)

Usage:
    python3 check_frozen_bitfit_mixup.py --task stsb
"""
import argparse
import hashlib
from pathlib import Path

import torch

METHODS = ["frozen", "bitfit", "full_ft"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_stable_rank_list(health_dict: dict, prefix: str = "V"):
    """Same logic as build_cache.py's extract_stable_rank_list."""
    target_key = "stable_rank_Weff_V" if prefix == "V" else "stable_rank_Weff_O"
    if target_key in health_dict and isinstance(health_dict[target_key], (list,)):
        return [float(x) for x in health_dict[target_key]]
    for k, v in health_dict.items():
        if prefix.lower() in k.lower() and "rank" in k.lower() and isinstance(v, list):
            return [float(x) for x in v]
    if "per_layer" in health_dict:
        layer_data = health_dict["per_layer"]
        sub_key = "W_V" if prefix == "V" else "W_O"
        rank_key = "V_stable_rank" if prefix == "V" else "O_stable_rank"
        try:
            if isinstance(layer_data, list):
                return [float(entry[sub_key][rank_key]) for entry in layer_data]
            if isinstance(layer_data, dict):
                return [float(layer_data[k][sub_key][rank_key])
                        for k in sorted(layer_data.keys(), key=int)]
        except (KeyError, TypeError):
            pass
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="stsb")
    p.add_argument("--results_dir", default="results/glue")
    args = p.parse_args()

    paths = {}
    for method in METHODS:
        method_dir = Path(args.results_dir) / args.task / method
        init_p = method_dir / "init" / "geometric_health.pt"
        final_p = method_dir / "final" / "geometric_health.pt"
        paths[method] = {"init": init_p, "final": final_p}

    # ── Part 1: file-level hash comparison ──────────────────────────────────
    print("=" * 70)
    print("PART 1: File hash comparison (detects literal file-level mixup)")
    print("=" * 70)
    hashes = {}
    for method in METHODS:
        for tag in ("init", "final"):
            path = paths[method][tag]
            if path.exists():
                h = file_hash(path)
                hashes[(method, tag)] = h
                print(f"  {method}/{tag}: {h[:16]}...  ({path})")
            else:
                print(f"  {method}/{tag}: FILE DOES NOT EXIST ({path})")

    print("\n  Cross-method comparisons (final vs final):")
    for i, m1 in enumerate(METHODS):
        for m2 in METHODS[i+1:]:
            h1 = hashes.get((m1, "final"))
            h2 = hashes.get((m2, "final"))
            if h1 and h2:
                same = "IDENTICAL FILE" if h1 == h2 else "different"
                print(f"    {m1}/final vs {m2}/final: {same}")

    for method in METHODS:
        h_init = hashes.get((method, "init"))
        h_final = hashes.get((method, "final"))
        if h_init and h_final:
            same = "IDENTICAL (expected if truly frozen)" if h_init == h_final else "DIFFERENT"
            print(f"    {method}: init vs final: {same}")

    # ── Part 2: direct per-layer stable rank extraction ─────────────────────
    print("\n" + "=" * 70)
    print("PART 2: Per-layer stable rank, extracted directly from raw files")
    print("=" * 70)
    for method in METHODS:
        print(f"\n-- {method} --")
        for tag in ("init", "final"):
            path = paths[method][tag]
            if not path.exists():
                print(f"  {tag}: file does not exist")
                continue
            health = torch.load(path, map_location="cpu", weights_only=True)
            sr_list = extract_stable_rank_list(health, "V")
            if sr_list:
                print(f"  {tag}: {[f'{v:.4f}' for v in sr_list]}")
            else:
                print(f"  {tag}: extract_stable_rank_list returned EMPTY "
                      f"(keys in file: {list(health.keys())})")

    print("\n" + "=" * 70)
    print("VERDICT GUIDE:")
    print("  If final-vs-final hashes show 'IDENTICAL FILE' between methods:")
    print("    -> literal file mixup (copy/symlink bug in training/checkpoint saving)")
    print("  If init vs final differs for frozen/bitfit despite files NOT being")
    print("  identical across methods:")
    print("    -> something genuinely wrote different data into 'final' than")
    print("       'init' for a method that should never change -- check the")
    print("       training/checkpoint-saving code for these methods specifically")
    print("  If everything here is clean (init==final for each method, files")
    print("  differ appropriately across methods):")
    print("    -> the raw data is fine, bug is downstream in build_cache.py or")
    print("       dump_geometric_heatmap_data.py -- needs a different check")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
analysis/recover_lora_weights.py

Recovers correct W_eff for LoRA runs by merging the adapters into the base model,
extracting head-structured tensor shapes, and caching them with keys that align
perfectly with standard PAFT metric tracking schemas.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import torch
from analysis.stable_rank import stable_rank as sr

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

_N_LAYERS = 12
_N_HEADS  = 12
_HEAD_DIM = 64
_HID_SIZE = 768

_TASK_NUM_LABELS = {
    "cola": 2, "mnli": 3, "mrpc": 2, "qnli": 2,
    "qqp":  2, "rte":  2, "sst2": 2, "stsb": 1,
}

LORA_METHODS = {"lora_r8", "lora_r64"}


def _find_best_checkpoint(hf_dir: Path) -> Path | None:
    ckpts = sorted(hf_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def _extract_deberta_weights(merged_model) -> dict:
    W_V_layers, W_O_layers = [], []
    deberta = merged_model.deberta

    for l in range(_N_LAYERS):
        attn_self = deberta.encoder.layer[l].attention.self
        attn_out  = deberta.encoder.layer[l].attention.output

        vp = attn_self.value_proj.weight.detach().float()
        W_V = (vp.reshape(_N_HEADS, _HEAD_DIM, _HID_SIZE)
                 .permute(0, 2, 1)
                 .contiguous().cpu())

        od = attn_out.dense.weight.detach().float()
        W_O = (od.reshape(_HID_SIZE, _N_HEADS, _HEAD_DIM)
                 .permute(1, 2, 0)
                 .contiguous().cpu())

        W_V_layers.append(W_V)
        W_O_layers.append(W_O)

    return {"W_V": W_V_layers, "W_O": W_O_layers}


def _compute_geometric_health(adapted: dict) -> dict:
    W_V_layers = adapted["W_V"]
    W_O_layers = adapted["W_O"]
    n_layers   = len(W_V_layers)

    stable_rank_Weff_V = []
    stable_rank_Weff_O = []
    per_layer = {}

    for l in range(n_layers):
        W_V_l = W_V_layers[l].float()  # [H, n, d]
        W_O_l = W_O_layers[l].float()  # [H, d, n]

        sr_V  = [sr(W_V_l[h]) for h in range(_N_HEADS)]
        sr_O  = [sr(W_O_l[h]) for h in range(_N_HEADS)]

        mean_V = sum(sr_V) / len(sr_V)
        mean_O = sum(sr_O) / len(sr_O)

        stable_rank_Weff_V.append(mean_V)
        stable_rank_Weff_O.append(mean_O)

        per_layer[l] = {
            "W_V": {"V_stable_rank": mean_V, "head_stable_ranks": sr_V},
            "W_O": {"O_stable_rank": mean_O, "head_stable_ranks": sr_O}
        }

    return {
        "stable_rank_Weff_V": stable_rank_Weff_V,
        "stable_rank_Weff_O": stable_rank_Weff_O,
        "per_layer_breakdown": per_layer,
        "global": {
            "W_V": {"V_stable_rank": sum(stable_rank_Weff_V) / n_layers},
            "W_O": {"O_stable_rank": sum(stable_rank_Weff_O) / n_layers},
        }
    }


def recover_run(run_dir: Path, task: str, force: bool = False) -> bool:
    merged_path = run_dir / "final" / "adapted_weights_merged.pt"
    if merged_path.exists() and not force:
        log.info(f"  SKIP (already recovered): {run_dir.name} (use --force to recompute)")
        return True

    hf_dir = run_dir / "hf_checkpoints"
    if not hf_dir.exists():
        log.warning(f"  NO hf_checkpoints found: {run_dir}")
        return False

    ckpt_dir = _find_best_checkpoint(hf_dir)
    if ckpt_dir is None:
        log.warning(f"  NO checkpoints in: {hf_dir}")
        return False

    num_labels = _TASK_NUM_LABELS.get(task, 2)
    log.info(f"  Loading {ckpt_dir.name} for {task}/{run_dir.name} ...")

    try:
        from transformers import AutoModelForSequenceClassification
        from peft import PeftModel

        base = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        peft_model = PeftModel.from_pretrained(base, str(ckpt_dir))
        merged = peft_model.merge_and_unload()

        merged.eval()

        adapted   = _extract_deberta_weights(merged)
        geo_health = _compute_geometric_health(adapted)

        torch.save(adapted,    merged_path)
        torch.save(geo_health, run_dir / "final" / "geometric_health_merged.pt")

        global_sr = geo_health["global"]["W_V"]["V_stable_rank"]
        log.info(f"  OK  sr(W_V) = {global_sr:.3f}  → {merged_path.name}")

        del merged, peft_model, base
        return True

    except Exception as e:
        log.error(f"  FAILED: {e}")
        return False


def main():
    from analysis.utils import setup_run_log
    setup_run_log("recover_lora_weight")

    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results", type=Path)
    p.add_argument(
        "--force", action="store_true",
        help="Recompute and overwrite adapted_weights_merged.pt / "
             "geometric_health_merged.pt even if they already exist. Needed "
             "after any fix to the merge logic, since existing output files "
             "otherwise silently short-circuit recomputation.",
    )
    args = p.parse_args()

    results_dir = args.results_dir
    ok, fail = 0, 0

    for task_dir in sorted((results_dir / "glue").iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name

        for method_dir in sorted(task_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            if method not in LORA_METHODS:
                continue
            if not (method_dir / "final" / "training_complete").exists():
                continue

            log.info(f"Recovering {task}/{method}")
            if recover_run(method_dir, task, force=args.force):
                ok += 1
            else:
                fail += 1

    log.info(f"\nDone. Recovered: {ok}  Failed: {fail}")


if __name__ == "__main__":
    main()
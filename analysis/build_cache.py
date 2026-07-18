#!/usr/bin/env python3
"""
analysis/build_cache.py

Aggregates parameters, metrics, and tensor dimensions across all finished
GLUE tasks into a centralized JSON data sheet. Introspects saved PyTorch
dictionaries dynamically to handle changing metrics keys without fallbacks.
"""
import json
import argparse
import logging
import time
from pathlib import Path
import torch
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

# In build_cache.py:
TASK_PRIMARY = {
    "cola": "matthews_correlation", "mnli": "accuracy", "mrpc": "f1",
    "qnli": "accuracy", "qqp": "f1", "rte": "accuracy",
    "sst2": "accuracy", "stsb": "pearson"
}


def extract_stable_rank_list(health_dict: dict, prefix: str = "V") -> list:
    """Introspects the metric dictionary to extract layer arrays safely."""
    # Pattern A: Flat dictionary mapping direct strings
    target_key = "stable_rank_Weff_V" if prefix == "V" else "stable_rank_Weff_O"
    if target_key in health_dict and isinstance(health_dict[target_key], (list, np.ndarray, torch.Tensor)):
        return [float(x) for x in health_dict[target_key]]

    # Pattern B: Match lowercase or variant keys
    for k, v in health_dict.items():
        if prefix.lower() in k.lower() and "rank" in k.lower() and isinstance(v, (list, np.ndarray)):
            return [float(x) for x in v]

    # Pattern C: per_layer is a list of dicts, each with W_V/W_O sub-dicts.
    # This is the actual structure written by the training code.
    if "per_layer" in health_dict:
        layer_data = health_dict["per_layer"]
        sub_key  = "W_V" if prefix == "V" else "W_O"
        rank_key = "V_stable_rank" if prefix == "V" else "O_stable_rank"
        try:
            if isinstance(layer_data, list):
                return [float(entry[sub_key][rank_key]) for entry in layer_data]
            # Fallback: dict keyed by layer index (older checkpoint format)
            if isinstance(layer_data, dict):
                return [float(layer_data[k][sub_key][rank_key])
                        for k in sorted(layer_data.keys(), key=int)]
        except (KeyError, TypeError):
            pass

    # Absolute structural backup if everything else is missing
    return []


def compute_metrics_from_weight(W: torch.Tensor):
    if W.dim() > 2:
        W = W.reshape(-1, W.shape[-1])
    W = W.float().cpu().numpy()

    try:
        _, s, _ = np.linalg.svd(W, full_matrices=False)
        squared_s = s ** 2
        sum_squared_s = np.sum(squared_s)

        if sum_squared_s < 1e-12:
            # ΔW is the zero matrix — sr is undefined, not a real value
            return None, None, None, None
        sr = float(sum_squared_s / (s[0] ** 2))

        p = squared_s / sum_squared_s if sum_squared_s > 0 else np.ones_like(s)
        p = p[p > 0]
        spectral_entropy = float(-np.sum(p * np.log(p)))
        eff_rank = float(np.exp(spectral_entropy))
        cond_num = float(s[0] / s[-1]) if s[-1] > 0 else 1.0

        return sr, spectral_entropy, eff_rank, cond_num
    except Exception as e:
        log.warning(f"SVD failed on weight matrix shape {W.shape}: {e} — skipping this weight")
        return None, None, None, None


def per_head_layer_metrics(W_init_layer: torch.Tensor, W_final_layer: torch.Tensor):
    """
    Compute sr(deltaW_h), spectral entropy, effective rank, and condition
    number via full SVD on EACH head's slice independently, then average
    across heads to a single per-layer value.

    This must match the paper's stated methodology exactly: "For each head
    h at layer l, all metrics are computed via full SVD on the per-head
    slices... Results are averaged across heads to yield per-layer values."
    Flattening all heads into one [H*n, d] matrix and running a single SVD
    (the previous behavior here) computes a genuinely different quantity —
    the spectral properties of a stacked matrix are not the same as the
    average of per-head spectral properties — so it must not be used for
    any metric this paper reports as "per-head."

    W_init_layer, W_final_layer: expected shape [H, n, d]. If a checkpoint's
    saved tensor has no explicit head dimension, falls back to a single SVD
    on the whole layer (old behavior) rather than crashing on an
    unexpected shape — this should not happen for W_V/W_O given the
    documented [H, n, d] snapshot format, but is defensive in case a
    future method's checkpoint format differs.

    Returns (sr_delta, entropy, eff_rank, cond) — each None if no head (or
    the fallback path) produced a valid value.
    """
    if W_init_layer.dim() < 3:
        log.warning("per_head_layer_metrics received a tensor with no head "
                    f"dimension (shape {tuple(W_init_layer.shape)}) — falling "
                    f"back to a single whole-layer SVD for this layer, not "
                    f"true per-head averaging.")
        dW = (W_final_layer - W_init_layer).reshape(-1, W_init_layer.shape[-1])
        sr_dw, _, _, _ = compute_metrics_from_weight(dW)
        _, ent, er, cond = compute_metrics_from_weight(
            W_final_layer.reshape(-1, W_final_layer.shape[-1])
        )
        return sr_dw, ent, er, cond

    n_heads = W_init_layer.shape[0]
    head_sr_delta, head_ent, head_er, head_cond = [], [], [], []

    for h in range(n_heads):
        Wh_init  = W_init_layer[h]
        Wh_final = W_final_layer[h]
        dWh = Wh_final - Wh_init

        sr_dw_h, _, _, _ = compute_metrics_from_weight(dWh)
        if sr_dw_h is not None:
            head_sr_delta.append(sr_dw_h)

        _, ent_h, er_h, cond_h = compute_metrics_from_weight(Wh_final)
        if ent_h is not None:
            head_ent.append(ent_h)
        if er_h is not None:
            head_er.append(er_h)
        if cond_h is not None:
            head_cond.append(cond_h)

    sr_delta = float(np.mean(head_sr_delta)) if head_sr_delta else None
    entropy  = float(np.mean(head_ent))      if head_ent      else None
    eff_rank = float(np.mean(head_er))       if head_er       else None
    cond     = float(np.mean(head_cond))     if head_cond     else None
    return sr_delta, entropy, eff_rank, cond


def main():
    from analysis.utils import setup_run_log
    setup_run_log("build_cache")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/glue", type=Path)
    parser.add_argument("--output_cache", default="results/analysis/metrics_cache.json", type=Path)
    args = parser.parse_args()

    root = args.results_dir
    out_cache = args.output_cache
    out_cache.parent.mkdir(parents=True, exist_ok=True)

    cache = {"glue": {}}
    tasks = [d.name for d in root.iterdir() if d.is_dir() if d.name in TASK_PRIMARY]
    log.info(f"Found {len(tasks)} tasks: {tasks}")

    for task_idx, task in enumerate(tasks):
        cache["glue"][task] = {}
        task_dir = root / task
        methods = [m.name for m in task_dir.iterdir() if m.is_dir()]
        log.info(f"[{task_idx+1}/{len(tasks)}] {task}: {len(methods)} methods "
                 f"— this involves per-head SVD (12 heads x V+O x delta+final "
                 f"per layer), which is slower than a single flattened SVD; "
                 f"expect real wall-clock time here, not a hang.")

        for method_idx, method in enumerate(methods):
            method_dir = task_dir / method
            metrics_file = method_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            log.info(f"  [{method_idx+1}/{len(methods)}] {task}/{method} ...")
            _method_start_time = time.time()

            with open(metrics_file) as f:
                m_data = json.load(f)

            primary_metric = TASK_PRIMARY[task]
            task_score = m_data.get(primary_metric, None)
            if task_score is None:
                continue

            is_lora = "lora" in method
            health_suffix = "_merged.pt" if is_lora else ".pt"
            weights_suffix = "_merged.pt" if is_lora else ".pt"

            init_health_p = method_dir / "init" / "geometric_health.pt"
            final_health_p = method_dir / "final" / f"geometric_health{health_suffix}"
            init_weights_p = method_dir / "init" / "adapted_weights.pt"
            final_weights_p = method_dir / "final" / f"adapted_weights{weights_suffix}"

            if not (init_health_p.exists() and final_health_p.exists()):
                continue

            init_health = torch.load(init_health_p, map_location="cpu")
            final_health = torch.load(final_health_p, map_location="cpu")

            # Use deep introspection instead of rigid key lookups
            list_init_v = extract_stable_rank_list(init_health, "V")
            list_final_v = extract_stable_rank_list(final_health, "V")

            if not list_init_v:
                log.warning(f"SKIP {task}/{method}: extract_stable_rank_list returned empty "
                            f"for init/geometric_health.pt — structure does not match any "
                            f"expected pattern. Check what keys are inside that file.")
                continue
            if not list_final_v:
                log.warning(f"SKIP {task}/{method}: extract_stable_rank_list returned empty "
                            f"for final/geometric_health{health_suffix} — structure mismatch "
                            f"or file missing.")
                continue

            # W_O — supplementary, not required. Every method has some
            # relationship to W_V (real change or genuine untouched
            # baseline), which is why it stays the primary cross-method
            # metric. W_O is only meaningful for methods that actually
            # target the full OV circuit; for others it's a real, correctly
            # -computed near-zero delta, not missing data. Either way this
            # must never block the entry — log and continue with None.
            list_init_o = extract_stable_rank_list(init_health, "O")
            list_final_o = extract_stable_rank_list(final_health, "O")
            sr_init_o = float(np.mean(list_init_o)) if list_init_o else None
            sr_final_o = float(np.mean(list_final_o)) if list_final_o else None
            if sr_init_o is None or sr_final_o is None:
                log.info(f"{task}/{method}: no W_O stable-rank data in geometric_health "
                         f"file(s) — sr_Weff_O_init/final will be null for this entry.")

            sr_init_v = float(np.mean(list_init_v))
            sr_final_v = float(np.mean(list_final_v))

            # These start as None, not placeholder numbers. If the block below
            # doesn't finish (missing weight files, or an exception), they
            # MUST stay None so downstream tables/plots render "---" instead
            # of a fabricated value that looks like real data.
            sr_delta_v, entropy_v, eff_rank_v, cond_v = None, None, None, None
            sr_delta_o, entropy_o, eff_rank_o, cond_o = None, None, None, None
            per_layer_records = []

            if init_weights_p.exists() and final_weights_p.exists():
                try:
                    w_i = torch.load(init_weights_p, map_location="cpu")
                    w_f = torch.load(final_weights_p, map_location="cpu")

                    delta_ranks = []
                    entropies = []
                    eff_ranks = []
                    cond_numbers = []

                    delta_ranks_o = []
                    entropies_o = []
                    eff_ranks_o = []
                    cond_numbers_o = []
                    has_o = "W_O" in w_i and "W_O" in w_f

                    for layer in range(len(w_i["W_V"])):
                        W_init = w_i["W_V"][layer]    # [H, n, d]
                        W_final = w_f["W_V"][layer]

                        sr_dw, ent_layer, er_layer, cond_layer = per_head_layer_metrics(
                            W_init, W_final
                        )
                        # None = zero matrix (frozen/bitfit) OR SVD failure (logged above)
                        delta_ranks.append(sr_dw)
                        if ent_layer is not None:
                            entropies.append(ent_layer)
                        if er_layer is not None:
                            eff_ranks.append(er_layer)
                        if cond_layer is not None:
                            cond_numbers.append(cond_layer)

                        layer_final_sr = list_final_v[layer] if layer < len(list_final_v) else sr_final_v
                        layer_init_sr = list_init_v[layer] if layer < len(list_init_v) else sr_init_v
                        layer_record = {
                            "layer": layer,
                            "sr_Weff_init": layer_init_sr,
                            "sr_Weff_final": layer_final_sr,
                            "sr_deltaW": sr_dw
                        }

                        # W_O — same per-head computation, only if this
                        # checkpoint actually saved W_O (it should, per the
                        # OV snapshot policy; not every method's snapshot
                        # may have it)
                        if has_o and layer < len(w_i["W_O"]) and layer < len(w_f["W_O"]):
                            WO_init = w_i["W_O"][layer]
                            WO_final = w_f["W_O"][layer]

                            sr_dw_o, ent_layer_o, er_layer_o, cond_layer_o = per_head_layer_metrics(
                                WO_init, WO_final
                            )
                            delta_ranks_o.append(sr_dw_o)
                            if ent_layer_o is not None:
                                entropies_o.append(ent_layer_o)
                            if er_layer_o is not None:
                                eff_ranks_o.append(er_layer_o)
                            if cond_layer_o is not None:
                                cond_numbers_o.append(cond_layer_o)

                            layer_final_sr_o = list_final_o[layer] if list_final_o and layer < len(list_final_o) else None
                            layer_record["sr_Weff_O_final"] = layer_final_sr_o
                            layer_record["sr_deltaW_O"] = sr_dw_o

                        per_layer_records.append(layer_record)

                    valid_deltas = [v for v in delta_ranks if v is not None]
                    sr_delta_v = float(np.mean(valid_deltas)) if valid_deltas else None
                    entropy_v = float(np.mean(entropies)) if entropies else None
                    eff_rank_v = float(np.mean(eff_ranks)) if eff_ranks else None
                    cond_v = float(np.mean(cond_numbers)) if cond_numbers else None

                    if has_o:
                        valid_deltas_o = [v for v in delta_ranks_o if v is not None]
                        sr_delta_o = float(np.mean(valid_deltas_o)) if valid_deltas_o else None
                        entropy_o = float(np.mean(entropies_o)) if entropies_o else None
                        eff_rank_o = float(np.mean(eff_ranks_o)) if eff_ranks_o else None
                        cond_o = float(np.mean(cond_numbers_o)) if cond_numbers_o else None
                    else:
                        log.info(f"{task}/{method}: adapted_weights file(s) have no 'W_O' key "
                                 f"— sr_deltaW_O/entropy_O/eff_rank_O/cond_O will be null.")

                except Exception as ex:
                    log.error(f"Error parsing layers for {task}/{method}: {ex} — "
                              f"sr_deltaW_V/spectral_entropy/effective_rank/condition_number "
                              f"will be written as null for this entry, not a placeholder value.")
            else:
                log.warning(f"{task}/{method}: adapted_weights.pt not found at init or final — "
                            f"sr_deltaW_V/spectral_entropy/effective_rank/condition_number "
                            f"will be written as null for this entry.")

            # Map structural dynamics over training iterations
            epoch_series = []
            epoch_dirs = sorted(list(method_dir.glob("epoch_*")), key=lambda x: int(x.name.split("_")[-1]))
            for ep_dir in epoch_dirs:
                ep_idx = int(ep_dir.name.split("_")[-1])
                ep_health_p = ep_dir / "geometric_health.pt"
                if ep_health_p.exists():
                    try:
                        ep_h = torch.load(ep_health_p, map_location="cpu")
                        ep_list = extract_stable_rank_list(ep_h, "V")
                        if ep_list:
                            ep_sr_val = float(np.mean(ep_list))
                            epoch_series.append({
                                "epoch": ep_idx,
                                "sr_Weff": ep_sr_val
                            })
                        else:
                            log.warning(f"{task}/{method} epoch {ep_idx}: no stable-rank "
                                        f"data found in {ep_health_p.name} — omitting this "
                                        f"epoch from per_epoch (NOT substituting sr_Weff_final)")
                    except Exception as ex:
                        log.error(f"{task}/{method} epoch {ep_idx}: failed to load/parse "
                                  f"{ep_health_p.name}: {ex} — omitting this epoch")

            def _sf(v):
                """Safe float: None stays None instead of crashing or being coerced."""
                return float(v) if v is not None else None

            cache["glue"][task][method] = {
                "task_score": float(task_score),
                "metric": primary_metric,
                "trainable_params": int(m_data.get("trainable_params", 0)),
                "sr_Weff_init": float(sr_init_v),
                "sr_Weff_final": float(sr_final_v),
                "sr_deltaW_V": _sf(sr_delta_v),
                "spectral_entropy_Weff_final": _sf(entropy_v),
                "effective_rank_Weff_final": _sf(eff_rank_v),
                "condition_number_final": _sf(cond_v),
                # W_O — supplementary. null unless this checkpoint actually
                # saved W_O and it was extracted successfully above.
                "sr_Weff_O_init": _sf(sr_init_o),
                "sr_Weff_O_final": _sf(sr_final_o),
                "sr_deltaW_O": _sf(sr_delta_o),
                "spectral_entropy_Weff_O_final": _sf(entropy_o),
                "effective_rank_Weff_O_final": _sf(eff_rank_o),
                "condition_number_O_final": _sf(cond_o),
                "per_epoch": epoch_series,
                "per_layer": per_layer_records
            }
            log.info(f"  [{method_idx+1}/{len(methods)}] {task}/{method} done "
                     f"in {time.time() - _method_start_time:.1f}s")

    with open(out_cache, "w") as f:
        json.dump(cache, f, indent=2)
    log.info(f"Introspective metrics cache compiled successfully -> {out_cache}")


if __name__ == "__main__":
    main()
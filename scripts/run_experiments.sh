#!/bin/bash
# run_experiments.sh
# Orchestration for all PAFT benchmark experiments.
#
# USAGE:
#   bash run_experiments.sh [--task-set glue|commonsense|gsm8k|all] \
#                           [--method METHOD] \
#                           [--dry-run] \
#                           [--resume]
#
# TOTAL RUNS:
#   GLUE:         8 tasks  × 8 methods = 64 runs  (~25 hours)
#   Commonsense:  8 tasks  × 6 methods = 48 runs  (~80 hours)
#   GSM8K:        1 task   × 6 methods =  6 runs  (~14 hours)
#   Ablations:    which-weights ablation = 10 extra runs
#   Total: ~128 runs + 10 ablations
#
# VRAM REQUIREMENTS:
#   GLUE (DeBERTa-v3-base, fp16):  ~1.5 GB  — any GPU
#   LLM  (LLaMA-3.2-3B, NF4):     ~4.0 GB  — 8 GB GPU minimum
#
# ENVIRONMENT:
#   Set LLAMA_MODEL to your local LLaMA path if you have it cached:
#   export LLAMA_MODEL=/path/to/llama-3.2-3b
#   Otherwise defaults to HuggingFace download.
#
# RESUME:
#   --resume flag skips any run where metrics.json already exists.
#   Safe to re-run after interruption.

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

RESULTS_DIR="${RESULTS_DIR:-results}"
LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.2-3B}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
TASK_SET="${TASK_SET:-all}"
FILTER_METHOD="${FILTER_METHOD:-}"
FILTER_TASK="${FILTER_TASK:-}"

# Parse flags
for arg in "$@"; do
    case $arg in
        --dry-run)      DRY_RUN=1 ;;
        --resume)       RESUME=1 ;;
        --task-set=*)   TASK_SET="${arg#*=}" ;;
        --method=*)     FILTER_METHOD="${arg#*=}" ;;
        --task=*)       FILTER_TASK="${arg#*=}" ;;
        --results=*)    RESULTS_DIR="${arg#*=}" ;;
    esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }
separator() { echo "────────────────────────────────────────────────────────"; }

# ────────────────────────────────────────────────────────────────────────────
# Task and method definitions
# ────────────────────────────────────────────────────────────────────────────

GLUE_TASKS="cola mnli mrpc qnli qqp rte sst2 stsb"
GLUE_METHODS="pure_paft hybrid_paft lora_r8 lora_r64 polar_r8 bitfit frozen full_ft svf"

CS_TASKS="boolq piqa siqa hellaswag winogrande arc_easy arc_challenge openbookqa"
LLM_METHODS="pure_paft hybrid_paft polar_r8 lora_r8 lora_r64 bitfit frozen"

GSM8K_TASK="gsm8k"

# Which-weights ablation (GLUE, DeBERTa)
# Tests: value_only, output_only, value+output (standard), query+value, all_four
ABLATION_METHODS="paft_value_only paft_output_only paft_query_value"
ABLATION_TASKS="rte sst2"   # two representative GLUE tasks

# ────────────────────────────────────────────────────────────────────────────
# Run helpers
# ────────────────────────────────────────────────────────────────────────────

should_skip() {
    local metrics_file="$1"
    if [ "$RESUME" = "1" ] && [ -f "$metrics_file" ]; then
        log "SKIP (already complete): $metrics_file"
        return 0
    fi
    return 1
}

run_cmd() {
    local cmd="$1"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY-RUN] $cmd"
    else
        eval "$cmd"
    fi
}

check_gpu() {
    if ! python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" 2>/dev/null; then
        log "WARNING: No CUDA GPU detected. GLUE runs will be slow; LLM runs may fail."
    else
        local vram
        vram=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')" 2>/dev/null || echo "unknown")
        log "GPU available: $(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown") — VRAM: $vram"
    fi
}

# ────────────────────────────────────────────────────────────────────────────
# GLUE experiments (DeBERTa-v3-base, fp16)
# ────────────────────────────────────────────────────────────────────────────

run_glue_experiments() {
    separator
    log "GLUE EXPERIMENTS: 8 tasks × 8 methods = 64 runs"
    separator

    local methods="$GLUE_METHODS"
    if [ -n "$FILTER_METHOD" ]; then methods="$FILTER_METHOD"; fi

    local total=0 skipped=0 failed=0

    for task in $GLUE_TASKS; do
        if [ -n "$FILTER_TASK" ] && [ "$task" != "$FILTER_TASK" ]; then continue; fi
        for method in $methods; do
            local out="$RESULTS_DIR/glue/$task/$method"
            total=$((total + 1))

            if should_skip "$out/metrics.json"; then
                skipped=$((skipped + 1))
                continue
            fi

            log "GLUE: task=$task  method=$method"
            local cmd="python -m paft.train_glue \
                --task $task \
                --method $method \
                --output_dir $out \
                --seed $SEED \
                --fp16"

            if ! run_cmd "$cmd"; then
                log "ERROR: $task/$method failed"
                failed=$((failed + 1))
            fi
        done
    done

    separator
    log "GLUE complete: total=$total  skipped=$skipped  failed=$failed"
}

# ────────────────────────────────────────────────────────────────────────────
# Commonsense reasoning experiments (LLaMA-3.2-3B, NF4)
# ────────────────────────────────────────────────────────────────────────────

run_commonsense_experiments() {
    separator
    log "COMMONSENSE EXPERIMENTS: 8 tasks × 6 methods = 48 runs"
    separator

    local methods="$LLM_METHODS"
    if [ -n "$FILTER_METHOD" ]; then methods="$FILTER_METHOD"; fi

    # Task-specific hyperparameters (matching PoLAR paper settings)
    declare -A TASK_LR=(
        [boolq]=3e-4    [piqa]=3e-4     [siqa]=3e-4
        [hellaswag]=3e-4 [winogrande]=3e-4
        [arc_easy]=3e-4  [arc_challenge]=3e-4 [openbookqa]=3e-4
    )
    declare -A TASK_EPOCHS=(
        [boolq]=3 [piqa]=3 [siqa]=3
        [hellaswag]=3 [winogrande]=3
        [arc_easy]=3 [arc_challenge]=3 [openbookqa]=3
    )

    local total=0 skipped=0 failed=0

    for task in $CS_TASKS; do
        if [ -n "$FILTER_TASK" ] && [ "$task" != "$FILTER_TASK" ]; then continue; fi
        for method in $methods; do
            local out="$RESULTS_DIR/commonsense/$task/$method"
            local lr="${TASK_LR[$task]:-3e-4}"
            local epochs="${TASK_EPOCHS[$task]:-3}"
            total=$((total + 1))

            if should_skip "$out/metrics.json"; then
                skipped=$((skipped + 1))
                continue
            fi

            log "Commonsense: task=$task  method=$method  lr=$lr"
            local cmd="python -m paft.train_llm \
                --task $task \
                --method $method \
                --model_name $LLAMA_MODEL \
                --output_dir $out \
                --epochs $epochs \
                --lr $lr \
                --batch_size 8 \
                --grad_accum 4 \
                --seed $SEED"

            if ! run_cmd "$cmd"; then
                log "ERROR: $task/$method failed"
                failed=$((failed + 1))
            fi
        done
    done

    separator
    log "Commonsense complete: total=$total  skipped=$skipped  failed=$failed"
}

# ────────────────────────────────────────────────────────────────────────────
# GSM8K experiments
# ────────────────────────────────────────────────────────────────────────────

run_gsm8k_experiments() {
    separator
    log "GSM8K EXPERIMENTS: 1 task × 6 methods = 6 runs"
    separator

    local methods="$LLM_METHODS"
    if [ -n "$FILTER_METHOD" ]; then methods="$FILTER_METHOD"; fi

    local total=0 skipped=0 failed=0

    for method in $methods; do
        local out="$RESULTS_DIR/gsm8k/$method"
        total=$((total + 1))

        if should_skip "$out/metrics.json"; then
            skipped=$((skipped + 1))
            continue
        fi

        log "GSM8K: method=$method"
        local cmd="python -m paft.train_llm \
            --task gsm8k \
            --method $method \
            --model_name $LLAMA_MODEL \
            --output_dir $out \
            --epochs 3 \
            --lr 3e-4 \
            --batch_size 4 \
            --grad_accum 8 \
            --use_metamath \
            --metamath_size 50000 \
            --seed $SEED"

        if ! run_cmd "$cmd"; then
            log "ERROR: gsm8k/$method failed"
            failed=$((failed + 1))
        fi
    done

    separator
    log "GSM8K complete: total=$total  skipped=$skipped  failed=$failed"
}

# ────────────────────────────────────────────────────────────────────────────
# Which-weights ablation (Analysis 5 in the paper)
# Equivalent to LoRA's Table 5 — fixed budget, vary which projections to adapt
# ────────────────────────────────────────────────────────────────────────────

run_ablation_experiments() {
    separator
    log "ABLATION: which-weights × 2 tasks = 10 runs (DeBERTa hybrid_paft)"
    separator

    # The ablation tests which attention projections benefit from PAFT most.
    # We run these as separate scripts with custom model configs.
    # NOTE: Requires ablation variants implemented in deberta_methods.py:
    #   paft_v_only, paft_o_only, paft_vo (standard), paft_qv, paft_all
    local ABLATION_VARIANTS="paft_v_only paft_o_only paft_qv paft_vo_qk paft_all"

    for task in $ABLATION_TASKS; do
        for variant in $ABLATION_VARIANTS; do
            local out="$RESULTS_DIR/ablation/which_weights/$task/$variant"
            if should_skip "$out/metrics.json"; then continue; fi

            log "Ablation: task=$task  variant=$variant"
            run_cmd "python -m paft.train_glue \
                --task $task \
                --method $variant \
                --output_dir $out \
                --seed $SEED"
        done
    done

    separator
    log "Ablation runs complete."
}

# ────────────────────────────────────────────────────────────────────────────
# Aggregate results into a summary table
# ────────────────────────────────────────────────────────────────────────────

collect_results() {
    separator
    log "Collecting results into summary tables ..."
    python - <<'EOF'
import json, os
from pathlib import Path

results = {}
root = Path(os.environ.get("RESULTS_DIR", "results"))

for metrics_file in sorted(root.rglob("metrics.json")):
    parts = metrics_file.parts
    try:
        benchmark = parts[-4]   # glue / commonsense / gsm8k
        task      = parts[-3]
        method    = parts[-2]
        with open(metrics_file) as f:
            m = json.load(f)
        key = f"{benchmark}/{task}/{method}"
        results[key] = m
    except (IndexError, json.JSONDecodeError):
        continue

# Print summary
print(f"\nResults Summary ({len(results)} runs complete)")
print("─" * 80)
for key, m in results.items():
    acc = m.get("accuracy") or m.get("matthews_correlation") or m.get("pearsonr") or m.get("f1")
    print(f"  {key:<50}  {acc:.4f}" if acc is not None else f"  {key}")

# Save to JSON
with open(root / "summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {root}/summary.json")
EOF
}

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

separator
log "PAFT Benchmark Experiments"
log "Task set: $TASK_SET  |  Results: $RESULTS_DIR  |  Dry-run: $DRY_RUN  |  Resume: $RESUME"
separator

check_gpu
mkdir -p "$RESULTS_DIR"

START_TIME=$(date +%s)

case "$TASK_SET" in
    glue)        run_glue_experiments ;;
    commonsense) run_commonsense_experiments ;;
    gsm8k)       run_gsm8k_experiments ;;
    ablation)    run_ablation_experiments ;;
    all)
        run_glue_experiments
        run_commonsense_experiments
        run_gsm8k_experiments
        run_ablation_experiments
        ;;
    *)
        echo "Unknown task set: $TASK_SET"
        echo "Choose from: glue commonsense gsm8k ablation all"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ "$DRY_RUN" = "0" ]; then
    collect_results
fi

separator
log "All experiments complete in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m ${ELAPSED}s"
log "Results directory: $RESULTS_DIR"
separator
#!/bin/bash
# run_experiments.sh

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results}"
LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.2-3B}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
TASK_SET="${TASK_SET:-all}"
FILTER_METHOD="${FILTER_METHOD:-}"
FILTER_TASK="${FILTER_TASK:-}"

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

GLUE_TASKS="cola mnli mrpc qnli qqp rte sst2 stsb"
GLUE_METHODS="pure_paft hybrid_paft safe_pure_paft safe_hybrid_paft lora_r8 lora_r64 polar_r8 bitfit svf full_ft frozen"

declare -A GLUE_LR=(
    [pure_paft]=5e-3
    [hybrid_paft]=3e-4
    [safe_pure_paft]=1e-3
    [safe_hybrid_paft]=3e-4
    [lora_r8]=4e-4
    [lora_r64]=3e-4
    [polar_r8]=1e-3
    [bitfit]=1e-3
    [svf]=1e-2
    [full_ft]=2e-5
    [frozen]=1e-4
)

# Bias LR for safe variants — used in the second AdamW parameter group.
# Geometric params (S/lam) use GLUE_LR; bias/classifier params use BIAS_LR.
BIAS_LR="1e-3"

CS_TASKS="boolq piqa siqa hellaswag winogrande arc_easy arc_challenge openbookqa"
LLM_METHODS="pure_paft hybrid_paft polar_r8 lora_r8 lora_r64 bitfit frozen"
GSM8K_TASK="gsm8k"
ABLATION_TASKS="rte sst2"

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
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log "WARNING: No CUDA GPU detected."
    else
        local vram
        vram=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')" 2>/dev/null || echo "unknown")
        log "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null) — VRAM: $vram"
    fi
}

run_glue_experiments() {
    separator
    log "GLUE EXPERIMENTS: 8 tasks × 11 methods = 88 runs"
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

            # Safe variants get a second LR for the bias parameter group.
            # All other methods pass --bias_lr but it is ignored by train_glue.py.
            local bias_lr_flag=""
            if [[ "$method" == safe_* ]]; then
                bias_lr_flag="--bias_lr $BIAS_LR"
                log "GLUE: task=$task  method=$method  geometric_lr=${GLUE_LR[$method]}  bias_lr=$BIAS_LR"
            else
                log "GLUE: task=$task  method=$method  lr=${GLUE_LR[$method]}"
            fi

            local cmd="python3 -m paft.training.train_glue \
                --task $task \
                --method $method \
                --lr ${GLUE_LR[$method]} \
                --max_length 128 \
                --output_dir $out \
                --seed $SEED \
                --no_fp16 \
                $bias_lr_flag"

            if ! run_cmd "$cmd"; then
                log "ERROR: $task/$method failed"
                failed=$((failed + 1))
            fi
        done
    done

    separator
    log "GLUE complete: total=$total  skipped=$skipped  failed=$failed"
}

run_commonsense_experiments() {
    separator
    log "COMMONSENSE EXPERIMENTS: 8 tasks × 7 methods = 56 runs"
    separator

    local methods="$LLM_METHODS"
    if [ -n "$FILTER_METHOD" ]; then methods="$FILTER_METHOD"; fi

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
            local cmd="python3 -m paft.training.train_llm \
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

run_gsm8k_experiments() {
    separator
    log "GSM8K EXPERIMENTS: 1 task × 7 methods = 7 runs"
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
        local cmd="python3 -m paft.training.train_llm \
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

run_ablation_experiments() {
    separator
    log "ABLATION: which-weights × 2 tasks"
    separator

    local ABLATION_VARIANTS="paft_v_only paft_o_only paft_qv paft_vo"

    for task in $ABLATION_TASKS; do
        for variant in $ABLATION_VARIANTS; do
            local out="$RESULTS_DIR/ablation/which_weights/$task/$variant"
            if should_skip "$out/metrics.json"; then continue; fi

            log "Ablation: task=$task  variant=$variant"
            run_cmd "python3 -m paft.training.train_glue \
                --task $task \
                --method $variant \
                --lr 3e-4 \
                --max_length 128 \
                --no_fp16 \
                --output_dir $out \
                --seed $SEED"
        done
    done

    separator
    log "Ablation runs complete."
}

collect_results() {
    separator
    log "Collecting results ..."
    python3 - <<'EOF'
import json, os
from pathlib import Path

# Primary metric per GLUE task — must match what train_glue.py reports
TASK_PRIMARY = {
    "cola":  "matthews_correlation",
    "stsb":  "pearson",
    "mrpc":  "f1",
    "qqp":   "f1",
    # All others: accuracy
}

results = {}
VALID_BENCHMARKS = {"glue", "commonsense", "gsm8k", "ablation"}
root = Path("results")

for metrics_file in sorted(root.rglob("metrics.json")):
    if "epoch_" in str(metrics_file) or "lr_sweep" in str(metrics_file):
        continue
    parts = metrics_file.parts
    if len(parts) >= 5:
        benchmark, task, method = parts[-4], parts[-3], parts[-2]
        if benchmark in VALID_BENCHMARKS:
            try:
                with open(metrics_file) as f:
                    m = json.load(f)
                # Use task-specific primary metric — avoids returning accuracy
                # when f1/mcc/pearson is the correct reporting metric
                primary = TASK_PRIMARY.get(task, "accuracy")
                score = (m.get(primary)
                         or m.get("accuracy")
                         or m.get("f1")
                         or m.get("matthews_correlation")
                         or m.get("pearson"))
                results[f"{benchmark}/{task}/{method}"] = {
                    "score": score,
                    "metric": primary,
                    **m
                }
            except Exception:
                continue

print(f"\nResults Summary ({len(results)} runs complete)")
print("─" * 80)
for key, m in results.items():
    score  = m.get("score")
    metric = m.get("metric", "?")
    print(f"  {key:<50}  {score:.4f}  [{metric}]" if score is not None else f"  {key}")

with open(root / "summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {root}/summary.json")
EOF
}

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
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
if [ "$DRY_RUN" = "0" ]; then collect_results; fi
separator
log "All experiments complete in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
separator
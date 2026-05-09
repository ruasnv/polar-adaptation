#!/bin/bash
# run_lm_eval.sh
# Step 2 of commonsense evaluation: export fine-tuned models and run lm-evaluation-harness.
#
# Run AFTER train_llm.py has completed for a task/method.
# Produces numbers directly comparable to PoLAR's published commonsense results.
#
# Prerequisites:
#   pip install lm-eval  (https://github.com/EleutherAI/lm-evaluation-harness)
#
# Usage:
#   bash run_lm_eval.sh                          # all tasks, all methods
#   bash run_lm_eval.sh --task=boolq             # one task
#   bash run_lm_eval.sh --task=boolq --method=pure_paft
#
# lm-eval task name mapping (HF dataset → lm-eval task name):
#   boolq       → boolq
#   piqa        → piqa
#   siqa        → social_iqa
#   hellaswag   → hellaswag
#   winogrande  → winogrande
#   arc_easy    → arc_easy
#   arc_challenge → arc_challenge
#   openbookqa  → openbookqa
#
# Results are saved to {RESULTS_DIR}/lm_eval/{task}/{method}/results.json

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results}"
EXPORT_DIR="${EXPORT_DIR:-exported_models}"
LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.2-3B}"
FILTER_TASK="${FILTER_TASK:-}"
FILTER_METHOD="${FILTER_METHOD:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"

for arg in "$@"; do
    case $arg in
        --task=*)    FILTER_TASK="${arg#*=}" ;;
        --method=*)  FILTER_METHOD="${arg#*=}" ;;
        --batch=*)   BATCH_SIZE="${arg#*=}" ;;
    esac
done

CS_TASKS="boolq piqa siqa hellaswag winogrande arc_easy arc_challenge openbookqa"
LLM_METHODS="pure_paft hybrid_paft polar_r8 lora_r8 lora_r64 bitfit frozen"

# Map our task names → lm-eval task names
declare -A LMEVAL_NAMES=(
    [boolq]=boolq
    [piqa]=piqa
    [siqa]=social_iqa
    [hellaswag]=hellaswag
    [winogrande]=winogrande
    [arc_easy]=arc_easy
    [arc_challenge]=arc_challenge
    [openbookqa]=openbookqa
)

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Check lm-eval is installed
if ! python -c "import lm_eval" 2>/dev/null; then
    echo "ERROR: lm-eval not installed. Run: pip install lm-eval"
    exit 1
fi

for task in $CS_TASKS; do
    if [ -n "$FILTER_TASK" ] && [ "$task" != "$FILTER_TASK" ]; then continue; fi
    lmeval_task="${LMEVAL_NAMES[$task]}"

    for method in $LLM_METHODS; do
        if [ -n "$FILTER_METHOD" ] && [ "$method" != "$FILTER_METHOD" ]; then continue; fi

        checkpoint_dir="$RESULTS_DIR/commonsense/$task/$method"
        export_path="$EXPORT_DIR/commonsense/$task/$method"
        result_path="$RESULTS_DIR/lm_eval/$task/$method"

        # Skip if lm-eval result already exists
        if [ -f "$result_path/results.json" ]; then
            log "SKIP (already evaluated): $task/$method"
            continue
        fi

        # Skip if training not complete
        if [ ! -f "$checkpoint_dir/final/training_complete" ]; then
            log "SKIP (training not complete): $task/$method"
            continue
        fi

        # Step 1: Export model
        if [ ! -d "$export_path" ]; then
            log "Exporting $task/$method ..."
            python -m paft.model.model_export \
                --checkpoint_dir "$checkpoint_dir" \
                --method "$method" \
                --model_name "$LLAMA_MODEL" \
                --output_dir "$export_path"
        else
            log "Export already exists: $task/$method"
        fi

        # Step 2: Run lm-evaluation-harness
        log "Running lm-eval: $task/$method (task=$lmeval_task) ..."
        mkdir -p "$result_path"

        python -m lm_eval \
            --model hf \
            --model_args "pretrained=$export_path,dtype=bfloat16" \
            --tasks "$lmeval_task" \
            --device cuda \
            --batch_size "$BATCH_SIZE" \
            --output_path "$result_path" \
            --log_samples

        log "Done: $task/$method"
        echo "  Result: $(python -c "
import json
with open('$result_path/results.json') as f: r = json.load(f)
results = r.get('results', {})
for k, v in results.items():
    acc = v.get('acc,none') or v.get('acc_norm,none') or 'N/A'
    print(f'  {k}: {acc:.4f}' if isinstance(acc, float) else f'  {k}: {acc}')
" 2>/dev/null || echo "see $result_path/results.json")"
    done
done

# Collect all lm-eval results into one summary table
python - << 'EOF'
import json, os
from pathlib import Path

results_root = Path(os.environ.get("RESULTS_DIR", "results")) / "lm_eval"
summary = {}

for results_file in sorted(results_root.rglob("results.json")):
    parts = results_file.parts
    try:
        task, method = parts[-3], parts[-2]
        with open(results_file) as f:
            data = json.load(f)
        r = data.get("results", {})
        for lmtask, metrics in r.items():
            acc = metrics.get("acc,none") or metrics.get("acc_norm,none")
            key = f"{task}/{method}"
            if key not in summary:
                summary[key] = {}
            summary[key]["accuracy"] = acc
    except (IndexError, json.JSONDecodeError, KeyError):
        continue

print(f"\nlm-evaluation-harness Results Summary ({len(summary)} runs)")
print("─" * 60)
for key, v in sorted(summary.items()):
    acc = v.get("accuracy")
    print(f"  {key:<50}  {acc:.4f}" if isinstance(acc, float) else f"  {key}")

out = results_root.parent / "lm_eval_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out}")
EOF
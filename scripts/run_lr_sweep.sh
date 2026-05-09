#!/bin/bash
# /scripts/run_lr_sweep.sh

# 1. PATH-PROOFING
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

RESULTS_DIR="results/lr_sweep"
TASK="sst2"
METHODS="pure_paft hybrid_paft lora_r8 lora_r64 polar_r8 bitfit svf"

mkdir -p "$RESULTS_DIR"

for method in $METHODS; do
    # 2. ASSIGN METHOD-SPECIFIC FAST RANGES
    # Focused on identifying the "basin of convergence" quickly.
    case $method in
        "pure_paft" | "svf")
            # Ultra-light methods need high LRs.
            LR_RANGE="1e-3 5e-3 1e-2"
            ;;
        "lora_r64" | "hybrid_paft")
            # High-parameter methods need stability.
            LR_RANGE="5e-5 1e-4 3e-4"
            ;;
        *)
            # Standard PEFT range for lora_r8, polar_r8, and bitfit.
            LR_RANGE="1e-4 4e-4 1e-3"
            ;;
    esac

    echo "--- Starting FAST sweep for $method (Range: $LR_RANGE) ---"

    for lr in $LR_RANGE; do
        OUT_DIR="$RESULTS_DIR/$method/lr_$lr"

        # Resume logic: skip if already complete
        if [ -f "$OUT_DIR/metrics.json" ]; then
            echo "Skipping $method at $lr (complete)"
            continue
        fi

        # 3. SPEED-RUN EXECUTION
        # Using 1 epoch and larger batch size for maximum throughput on RTX 5070.
        python3 -m paft.training.train_glue \
            --task "$TASK" \
            --method "$method" \
            --lr "$lr" \
            --epochs 1 \
            --output_dir "$OUT_DIR" \
            --batch_size 32 \
            --grad_accum 1 \
            --max_length 128 \
            --no_fp16
    done
done

# 4. IDENTIFY BEST LR
echo "--- Fast Sweep Complete. Identifying Best Learning Rates ---"
python3 -c "
import json, glob, os
methods = '$METHODS'.split()
results_root = '$RESULTS_DIR'
for method in methods:
    best_acc = 0; best_lr = None
    pattern = os.path.join(results_root, method, 'lr_*', 'metrics.json')
    for f in glob.glob(pattern):
        try:
            with open(f) as j: m = json.load(j)
            acc = m.get('accuracy', 0)
            lr = f.split(os.sep)[-2].replace('lr_', '')
            if acc > best_acc:
                best_acc = acc; best_lr = lr
        except: continue
    if best_lr:
        print(f'Method: {method:<15} | Best LR: {best_lr:<7} | Accuracy: {best_acc:.4f}')
    else:
        print(f'Method: {method:<15} | No results found.')
"
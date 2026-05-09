#!/bin/bash
# /scripts/run_lr_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

RESULTS_DIR="results/lr_sweep"
TASK="sst2"
METHODS="pure_paft hybrid_paft lora_r8 lora_r64 polar_r8 bitfit svf"

mkdir -p "$RESULTS_DIR"

for method in $METHODS; do
    # 1. ASSIGN METHOD-SPECIFIC RANGES
    # LoRA/PoLAR papers tune LRs separately because parameter density varies
    case $method in
        "pure_paft" | "svf")
            # Ultra-light methods (~18k params) need high LRs to move the needle
            LR_RANGE="5e-4 1e-3 3e-3 5e-3 1e-2"
            ;;
        "lora_r64" | "hybrid_paft")
            # High-parameter methods (1M-4M params) explode if LR is too high
            LR_RANGE="1e-5 5e-5 1e-4 3e-4"
            ;;
        *)
            # Standard PEFT range for lora_r8, polar_r8, and bitfit
            LR_RANGE="5e-5 1e-4 3e-4 1e-3"
            ;;
    esac

    echo "--- Starting sweep for $method (Range: $LR_RANGE) ---"

    for lr in $LR_RANGE; do
        OUT_DIR="$RESULTS_DIR/$method/lr_$lr"

        if [ -f "$OUT_DIR/metrics.json" ]; then
            echo "Skipping $method at $lr (complete)"
            continue
        fi

        # 2. RUN IN FULL FP32 PRECISION
        # No FP16, no NaNs. 8GB VRAM is enough for DeBERTa-v3 in FP32
        python3 -m paft.training.train_glue \
            --task "$TASK" \
            --method "$method" \
            --lr "$lr" \
            --epochs 3 \
            --output_dir "$OUT_DIR" \
            --batch_size 16 \
            --grad_accum 2 \
            --max_length 128 \
            --no_fp16
    done
done

# 3. IDENTIFY BEST LR (Remains the same)
echo "--- Sweep Complete. Identifying Best Learning Rates ---"
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
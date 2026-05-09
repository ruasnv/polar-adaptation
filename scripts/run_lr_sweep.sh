#!/bin/bash
# /scripts/run_lr_sweep.sh

# 1. PATH-PROOFING: Automatically find the project root
# This gets the directory where the script lives, then goes up one level.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 2. Move to the project root so 'python -m paft' and 'results/' work correctly
cd "$PROJECT_ROOT"

# 3. Ensure the project root is in PYTHONPATH for the 'paft' package
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

RESULTS_DIR="results/lr_sweep"
TASK="sst2"
# All methods that need LR tuning. full_ft and frozen excluded:
#   full_ft: well-established range (1e-5 to 5e-5), not worth sweeping
#   frozen:  no trainable params — LR irrelevant
METHODS="pure_paft hybrid_paft lora_r8 lora_r64 polar_r8 bitfit svf"

# Extended range: pure_paft and svf have very few params and often need 5e-3 to 1e-2
LR_CANDIDATES="1e-5 5e-5 1e-4 3e-4 1e-3 3e-3 5e-3 1e-2"

mkdir -p "$RESULTS_DIR"

for method in $METHODS; do
    echo "--- Starting sweep for method: $method ---"
    for lr in $LR_CANDIDATES; do
        OUT_DIR="$RESULTS_DIR/$method/lr_$lr"

        # Skip if already done
        if [ -f "$OUT_DIR/metrics.json" ]; then
            echo "Skipping $method at $lr (already complete)"
            continue
        fi

        python3 -m paft.training.train_glue \
            --task "$TASK" \
            --method "$method" \
            --lr "$lr" \
            --epochs 3 \
            --output_dir "$OUT_DIR" \
            --batch_size 16 \
            --grad_accum 2 \
            --max_length 128
    done
done

# 4. Analysis Block (Fixed variable passing)
echo "--- Sweep Complete. Identifying Best Learning Rates ---"
python -c "
import json, glob, os
methods = '$METHODS'.split()
results_root = '$RESULTS_DIR'
for method in methods:
    best_acc = 0; best_lr = None
    # Look for results in the specific method folder
    pattern = os.path.join(results_root, method, 'lr_*', 'metrics.json')
    for f in glob.glob(pattern):
        try:
            with open(f) as j: m = json.load(j)
            acc = m.get('accuracy', 0)
            # Extract LR from the directory name (e.g., .../lr_1e-3/metrics.json)
            lr = f.split(os.sep)[-2].replace('lr_', '')
            if acc > best_acc:
                best_acc = acc; best_lr = lr
        except: continue
    if best_lr:
        print(f'Method: {method:<15} | Best LR: {best_lr:<7} | Accuracy: {best_acc:.4f}')
    else:
        print(f'Method: {method:<15} | No results found.')
"
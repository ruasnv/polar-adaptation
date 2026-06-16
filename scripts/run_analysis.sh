#!/bin/bash
# ============================================================================
# run_analysis.sh
#
# Full analysis pipeline for PAFT experiments.
# Runs in correct dependency order:
#   1. Clear old analysis outputs
#   2. Patch stable_rank.py with new metrics (if not already present)
#   3. Recover LoRA merged weights from hf_checkpoints
#   4. Build metrics_cache.json (all methods)
#   5. Build paft_cache.json (PAFT-specific: Q drift, S asymmetry, lam shift)
#   6. Run all analysis and plotting scripts
#   7. Run dump_results.py (human-readable summary)
#
# Usage:
#   bash run_analysis.sh
#   bash run_analysis.sh --skip-clear     # keep old analysis outputs
#   bash run_analysis.sh --skip-lora      # skip LoRA weight recovery
#   bash run_analysis.sh --skip-cache     # skip cache rebuild (use existing)
#   bash run_analysis.sh --plots-only     # skip steps 1-5, only run plots
#
# Requirements:
#   All scripts live in analysis/ and are run as python -m analysis.<name>
#   Results dir: results/glue/
#   Output dir:  results/analysis/
#
#bash run_analysis.sh --skip-lora    # LoRA already recovered, skip step 3
#bash run_analysis.sh --skip-cache   # Cache already built, only run plots
#bash run_analysis.sh --plots-only   # Same as skip-clear + skip-lora + skip-cache
# ============================================================================

set -euo pipefail

# ── Flags ────────────────────────────────────────────────────────────────────
SKIP_CLEAR=0
SKIP_LORA=0
SKIP_CACHE=0
PLOTS_ONLY=0

for arg in "$@"; do
    case $arg in
        --skip-clear)  SKIP_CLEAR=1  ;;
        --skip-lora)   SKIP_LORA=1   ;;
        --skip-cache)  SKIP_CACHE=1  ;;
        --plots-only)  SKIP_CLEAR=1; SKIP_LORA=1; SKIP_CACHE=1; PLOTS_ONLY=0 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

RESULTS_DIR="results/glue"
ANALYSIS_DIR="results/analysis"
FIGURES_DIR="results/analysis/figures"

# ── Logging ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date +%H:%M:%S)] $*"; }
ok()   { echo "[$(date +%H:%M:%S)] ✓ $*"; }
warn() { echo "[$(date +%H:%M:%S)] ⚠ $*"; }
fail() { echo "[$(date +%H:%M:%S)] ✗ FAILED: $*" >&2; exit 1; }

sep() { echo "────────────────────────────────────────────────────"; }

run_py() {
    # Run a python module, exit on failure
    local module="$1"
    shift
    log "Running: python3 -m $module $*"
    if python3 -m "$module" "$@"; then
        ok "$module"
    else
        warn "$module FAILED — continuing with remaining scripts"
        return 1
    fi
}

# ── Step 0: Check environment ─────────────────────────────────────────────────
sep
log "PAFT Analysis Pipeline"
log "Project root: $PROJECT_ROOT"
sep

python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || \
    fail "PyTorch not available"

if [ ! -d "$RESULTS_DIR" ]; then
    fail "Results directory not found: $RESULTS_DIR — run experiments first"
fi

# ── Step 1: Clear old analysis outputs ────────────────────────────────────────
if [ "$SKIP_CLEAR" = "0" ]; then
    sep
    log "STEP 1: Clearing old analysis outputs"
    if [ -d "$ANALYSIS_DIR" ]; then
        log "  Removing $ANALYSIS_DIR"
        rm -rf "$ANALYSIS_DIR"
    fi
    ok "Analysis directory cleared"
else
    log "STEP 1: Skipped (--skip-clear)"
fi

# Always ensure output dirs exist
mkdir -p "$ANALYSIS_DIR" "$FIGURES_DIR"

# ── Step 2: Patch stable_rank.py with new metrics ─────────────────────────────
sep
log "STEP 2: Checking stable_rank.py for required metric functions"

STABLE_RANK_PATH="analysis/stable_rank.py"
if [ ! -f "$STABLE_RANK_PATH" ]; then
    fail "analysis/stable_rank.py not found — expected at $STABLE_RANK_PATH"
fi

# Check for each required function
MISSING_FUNCS=()
for func in "def isotropy" "def participation_ratio" "def nuclear_norm_ratio"; do
    if ! grep -q "$func" "$STABLE_RANK_PATH"; then
        MISSING_FUNCS+=("$func")
    fi
done

if [ ${#MISSING_FUNCS[@]} -gt 0 ]; then
    warn "Missing functions in stable_rank.py: ${MISSING_FUNCS[*]}"
    warn "These are needed for TABLE 6 isotropy/PR/NNRatio columns."
    warn "Add isotropy(), participation_ratio(), nuclear_norm_ratio() to stable_rank.py"
    warn "then rerun. Continuing without them — those columns will show N/A."
else
    ok "stable_rank.py has all required metric functions"
fi

# ── Step 3: Recover LoRA merged weights ───────────────────────────────────────
sep
if [ "$SKIP_LORA" = "0" ]; then
    log "STEP 3: Recovering LoRA merged weights from hf_checkpoints"
    log "  This loads each LoRA checkpoint, merges adapters, extracts per-head weights."
    log "  Takes ~15 minutes. Use --skip-lora if already done."

    # First delete any stale merged files from previous broken runs
    STALE_COUNT=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
    if [ "$STALE_COUNT" -gt 0 ]; then
        log "  Removing $STALE_COUNT stale merged weight files from previous runs"
        find "$RESULTS_DIR" -name "adapted_weights_merged.pt" -delete
        find "$RESULTS_DIR" -name "geometric_health_merged.pt" -delete
    fi

    run_py analysis.recover_lora_weight --results_dir results || \
        warn "LoRA recovery had errors — LoRA geometric data may be incomplete"

    # Verify recovery produced files
    RECOVERED=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
    log "  Recovered $RECOVERED merged weight files"
    if [ "$RECOVERED" -eq 0 ]; then
        warn "No merged weight files created — check hf_checkpoints exist under results/glue/"
        warn "LoRA methods will be excluded from geometric analysis"
    fi
else
    log "STEP 3: Skipped (--skip-lora)"
    RECOVERED=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
    log "  Existing merged files: $RECOVERED"
fi

# ── Step 4: Build metrics_cache.json ─────────────────────────────────────────
sep
if [ "$SKIP_CACHE" = "0" ]; then
    log "STEP 4: Building metrics_cache.json"
    log "  Reads adapted_weights.pt + geometric_health.pt for all methods/tasks."

    run_py analysis.build_cache \
        --results_dir "$RESULTS_DIR" \
        --output_cache "$ANALYSIS_DIR/metrics_cache.json" || \
        fail "build_cache failed — cannot continue without metrics_cache.json"

    # Sanity check: verify cache is not empty
    python3 -c "
import json
c = json.load(open('$ANALYSIS_DIR/metrics_cache.json'))
n_runs = sum(len(v) for v in c.get('glue', {}).values())
print(f'  Cache contains {n_runs} method-task entries')
assert n_runs > 0, 'Cache is empty!'
"   || fail "metrics_cache.json is empty or malformed"
    ok "metrics_cache.json built successfully"
else
    log "STEP 4: Skipped (--skip-cache)"
    if [ ! -f "$ANALYSIS_DIR/metrics_cache.json" ]; then
        fail "metrics_cache.json not found and --skip-cache was set"
    fi
fi

# ── Step 5: Build paft_cache.json ────────────────────────────────────────────
sep
if [ "$SKIP_CACHE" = "0" ]; then
    log "STEP 5: Building paft_cache.json"
    log "  Reads paft_snapshot.pt for Q drift, S asymmetry, eigenvalue shifts."

    run_py analysis.build_paft_cache \
        --results_dir "$RESULTS_DIR" \
        --output_cache "$ANALYSIS_DIR/paft_cache.json" || \
        warn "build_paft_cache had errors — PAFT-specific analyses may be incomplete"

    ok "paft_cache.json built"
else
    log "STEP 5: Skipped (--skip-cache)"
fi

# ── Step 6: Analysis scripts ──────────────────────────────────────────────────
sep
log "STEP 6: Running analysis scripts"
log "  Failed scripts are logged but do not stop the pipeline."
log "  Output: $FIGURES_DIR/"

PASSED=0
FAILED=0
FAILED_SCRIPTS=()

run_analysis() {
    local module="$1"
    shift
    if python3 -m "$module" "$@" 2>&1; then
        ok "$module"
        PASSED=$((PASSED + 1))
    else
        warn "$module FAILED"
        FAILED=$((FAILED + 1))
        FAILED_SCRIPTS+=("$module")
    fi
}

# ── Group A: No cache needed — read metrics.json directly ─────────────────────
log ""
log "  Group A: Performance tables (read metrics.json directly)"
run_analysis analysis.table_syntax_semantics
run_analysis analysis.plot_efficiency_curve

# ── Group B: Read metrics_cache.json ─────────────────────────────────────────
log ""
log "  Group B: Geometric analyses (read metrics_cache.json)"
run_analysis analysis.table_stable_rank
run_analysis analysis.table_all_metrics
run_analysis analysis.plot_sr_scatter
run_analysis analysis.table_correlation
run_analysis analysis.plot_layer_profile
run_analysis analysis.plot_layer_profiles_delta
run_analysis analysis.plot_training_dynamics
run_analysis analysis.plot_geometric_heatmaps

# ── Group C: Read paft_cache.json ────────────────────────────────────────────
log ""
log "  Group C: PAFT-specific analyses (read paft_cache.json)"
run_analysis analysis.plot_rotation_drift
run_analysis analysis.plot_eigenvalue_shift

# ── Group D: Human-readable dump ─────────────────────────────────────────────
log ""
log "  Group D: Human-readable results dump"
run_analysis analysis.dump_results \
    --cache "$ANALYSIS_DIR/metrics_cache.json" \
    --paft  "$ANALYSIS_DIR/paft_cache.json"

# ── Summary ───────────────────────────────────────────────────────────────────
sep
log "Pipeline complete."
log "  Passed: $PASSED"
log "  Failed: $FAILED"

if [ ${#FAILED_SCRIPTS[@]} -gt 0 ]; then
    warn "Failed scripts:"
    for s in "${FAILED_SCRIPTS[@]}"; do
        echo "    - $s"
    done
    warn "These scripts may not exist yet or may have bugs."
    warn "The cache files are valid — rerun individual scripts once fixed."
fi

log ""
log "Output files:"
log "  Cache:    $ANALYSIS_DIR/metrics_cache.json"
log "  PAFT:     $ANALYSIS_DIR/paft_cache.json"
log "  Figures:  $FIGURES_DIR/"
log "  Dump:     (printed above)"

# List generated figures
if [ -d "$FIGURES_DIR" ]; then
    NFIGS=$(find "$FIGURES_DIR" -name "*.pdf" -o -name "*.png" | wc -l)
    log "  Generated $NFIGS figure files"
    find "$FIGURES_DIR" -name "*.pdf" -o -name "*.png" | sort | \
        while read f; do echo "    $f"; done
fi
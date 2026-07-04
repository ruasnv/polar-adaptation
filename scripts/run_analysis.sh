#!/bin/bash
# ============================================================================
# run_analysis.sh
#
# Full analysis pipeline for PAFT experiments.
# Runs in correct dependency order:
#   1. Clear old analysis outputs
#   2. Recover LoRA merged weights  (skip with --skip-lora)
#   3. Build metrics_cache.json
#   4. Build paft_cache.json
#   5. Run all table scripts
#   6. Run all plotting scripts
#   7. Run dump_results (human-readable summary + ASCII figures, saved to txt)
#
# Usage:
#   bash run_analysis.sh                  # full run
#   bash run_analysis.sh --skip-lora      # LoRA already recovered
#   bash run_analysis.sh --skip-cache     # use existing cache, only run plots/tables
#   bash run_analysis.sh --skip-clear     # keep old analysis outputs
#   bash run_analysis.sh --plots-only     # skip steps 1-4, only run tables+plots+dump
# ============================================================================

set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_CLEAR=0
SKIP_LORA=0
SKIP_CACHE=0

for arg in "$@"; do
    case $arg in
        --skip-clear)  SKIP_CLEAR=1 ;;
        --skip-lora)   SKIP_LORA=1  ;;
        --skip-cache)  SKIP_CACHE=1 ;;
        --plots-only)  SKIP_CLEAR=1; SKIP_LORA=1; SKIP_CACHE=1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives in scripts/ — project root is one level up
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

RESULTS_DIR="results/glue"
ANALYSIS_DIR="results/analysis"
FIGURES_DIR="results/analysis/figures"
DUMP_TXT="$ANALYSIS_DIR/dump_results.txt"

# ── Logging ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date +%H:%M:%S)] $*"; }
ok()   { echo "[$(date +%H:%M:%S)] ✓ $*"; }
warn() { echo "[$(date +%H:%M:%S)] ⚠  $*"; }
fail() { echo "[$(date +%H:%M:%S)] ✗ FAILED: $*" >&2; exit 1; }
sep()  { echo "────────────────────────────────────────────────────"; }

PASSED=0
FAILED=0
FAILED_SCRIPTS=()

run_py() {
    # Run one python module. On failure: warn and continue (don't stop pipeline).
    local module="$1"; shift
    log "python3 -m $module $*"
    if python3 -m "$module" "$@" 2>&1; then
        ok "$module"
        PASSED=$((PASSED + 1))
    else
        warn "$module FAILED"
        FAILED=$((FAILED + 1))
        FAILED_SCRIPTS+=("$module")
    fi
}

# ── Step 0: Environment check ─────────────────────────────────────────────────
sep
log "PAFT Analysis Pipeline — $(date)"
log "Project root: $PROJECT_ROOT"
sep

python3 -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
" || fail "PyTorch not available in current environment"

[ -d "$RESULTS_DIR" ] || fail "Results directory not found: $RESULTS_DIR — run experiments first"

# ── Step 1: Clear old outputs ─────────────────────────────────────────────────
sep
if [ "$SKIP_CLEAR" = "0" ]; then
    log "STEP 1: Clearing old analysis outputs"
    [ -d "$ANALYSIS_DIR" ] && rm -rf "$ANALYSIS_DIR" && log "  Removed $ANALYSIS_DIR"
    ok "Cleared"
else
    log "STEP 1: Skipped (--skip-clear)"
fi
mkdir -p "$ANALYSIS_DIR" "$FIGURES_DIR"

# ── Step 2: LoRA weight recovery ──────────────────────────────────────────────
sep
if [ "$SKIP_LORA" = "0" ]; then
    log "STEP 2: Recovering LoRA merged weights"
    log "  Merges adapter weights into base model, extracts per-head W_eff."
    log "  ~15 min. Rerun with --skip-lora once complete."

    # Remove stale merged files from previous runs
    STALE=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
    if [ "$STALE" -gt 0 ]; then
        log "  Removing $STALE stale merged files"
        find "$RESULTS_DIR" -name "adapted_weights_merged.pt" -delete
        find "$RESULTS_DIR" -name "geometric_health_merged.pt" -delete
    fi

    if python3 -m analysis.recover_lora_weight --results_dir results 2>&1; then
        RECOVERED=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
        ok "LoRA recovery complete — $RECOVERED merged files created"
    else
        warn "LoRA recovery had errors — LoRA geometric data may be incomplete"
        RECOVERED=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
        log "  Merged files found: $RECOVERED"
    fi
else
    log "STEP 2: Skipped (--skip-lora)"
    RECOVERED=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
    log "  Existing merged files: $RECOVERED"
fi

# ── Step 3: Build metrics_cache.json ─────────────────────────────────────────
sep
if [ "$SKIP_CACHE" = "0" ]; then
    log "STEP 3: Building metrics_cache.json"
    python3 -m analysis.build_cache \
        --results_dir "$RESULTS_DIR" \
        --output_cache "$ANALYSIS_DIR/metrics_cache.json" 2>&1 || \
        fail "build_cache failed — cannot continue without metrics_cache.json"

    python3 -c "
import json
c = json.load(open('$ANALYSIS_DIR/metrics_cache.json'))
n = sum(len(v) for v in c.get('glue', {}).values())
print(f'  Cache: {n} method-task entries')
assert n > 0, 'Cache is empty'
" || fail "metrics_cache.json is empty or malformed"
    ok "metrics_cache.json"
else
    log "STEP 3: Skipped (--skip-cache)"
    [ -f "$ANALYSIS_DIR/metrics_cache.json" ] || \
        fail "metrics_cache.json not found — remove --skip-cache to rebuild"
fi

# ── Step 4: Build paft_cache.json ────────────────────────────────────────────
sep
if [ "$SKIP_CACHE" = "0" ]; then
    log "STEP 4: Building paft_cache.json"
    python3 -m analysis.build_paft_cache 2>&1 || \
        warn "build_paft_cache had errors — PAFT geometric analyses may be incomplete"
    ok "paft_cache.json"
else
    log "STEP 4: Skipped (--skip-cache)"
fi

# ── Step 5: Tables ────────────────────────────────────────────────────────────
sep
log "STEP 5: Generating tables → $ANALYSIS_DIR/"

run_py analysis.tables.table_syntax_semantics
run_py analysis.tables.table_stable_rank
run_py analysis.tables.table_all_metrics
run_py analysis.tables.table_correlation

# ── Step 6: Figures ───────────────────────────────────────────────────────────
sep
log "STEP 6: Generating figures → $FIGURES_DIR/"

run_py analysis.plotting.plot_efficiency_curve
run_py analysis.plotting.plot_sr_scatter
run_py analysis.plotting.plot_training_dynamics
run_py analysis.plotting.plot_layer_profiles_delta
run_py analysis.plotting.plot_rotation_drift
run_py analysis.plotting.plot_collapse
run_py analysis.plotting.plot_eigenvalue_shift
run_py analysis.plotting.plot_geometric_heatmaps

# ── Step 7: Dump results ──────────────────────────────────────────────────────
sep
log "STEP 7: Running dump_results → $DUMP_TXT"

python3 -m analysis.dump_results \
    --cache "$ANALYSIS_DIR/metrics_cache.json" \
    --paft  "$ANALYSIS_DIR/paft_cache.json" \
    --checkpoint_dir "$RESULTS_DIR" \
    2>&1 | tee "$DUMP_TXT"

ok "dump_results saved to $DUMP_TXT"

# ── Summary ───────────────────────────────────────────────────────────────────
sep
log "Pipeline complete — $(date)"
log "  Tables passed:  $PASSED"
log "  Scripts failed: $FAILED"

if [ "${#FAILED_SCRIPTS[@]}" -gt 0 ]; then
    warn "Failed scripts:"
    for s in "${FAILED_SCRIPTS[@]}"; do
        echo "      - $s"
    done
fi

log ""
log "Outputs:"
log "  Cache:       $ANALYSIS_DIR/metrics_cache.json"
log "  PAFT cache:  $ANALYSIS_DIR/paft_cache.json"
log "  Dump text:   $DUMP_TXT"
log "  Derived:     $ANALYSIS_DIR/derived_statistics.json"
log ""

if [ -d "$FIGURES_DIR" ]; then
    NFIGS=$(find "$FIGURES_DIR" -name "*.pdf" | wc -l)
    log "  Figures ($NFIGS):"
    find "$FIGURES_DIR" -name "*.pdf" | sort | while read -r f; do
        echo "      $f"
    done
fi
#!/bin/bash
# ============================================================================
# run_analysis.sh
#
# Full analysis pipeline for PAFT experiments.
# Runs in correct dependency order:
#   1. Clear old analysis outputs
#   2. Recover LoRA merged weights — final AND per-epoch  (skip with --skip-lora)
#   3. Build metrics_cache.json
#   4. Build paft_cache.json
#   5. Run all table scripts
#   6. Run all plotting scripts
#   7. Run dump_results (human-readable summary + ASCII figures, saved to txt)
#   8. LLaMA-3.2-3B pipeline (skip with --skip-llama):
#        8a. Recover LLaMA LoRA merged weights
#        8b. Collect LLaMA task-accuracy results
#        8c. Generate LLaMA geometric tables, Q-drift, asymmetry, analysis dump
#      Runs against results/llama, independent of the GLUE/DeBERTa steps above.
#      If results/llama doesn't exist, this step warns and skips rather than
#      failing the whole pipeline — the GLUE-only case is still valid.
#
# Usage:
#   bash run_analysis.sh                  # full run (GLUE + LLaMA)
#   bash run_analysis.sh --skip-lora      # LoRA already recovered
#   bash run_analysis.sh --skip-cache     # use existing cache, only run plots/tables
#   bash run_analysis.sh --skip-clear     # keep old analysis outputs
#   bash run_analysis.sh --skip-llama     # GLUE/DeBERTa only, skip Step 8 entirely
#   bash run_analysis.sh --plots-only     # skip steps 1-4, only run tables+plots+dump
#   bash run_analysis.sh --force          # pass --force to recovery/merge scripts,
#                                          # forcing recomputation even where output
#                                          # files already exist on disk. Use this
#                                          # any time a fix landed in
#                                          # recover_lora_weight.py,
#                                          # recover_lora_weights_llama.py,
#                                          # compute_lora_epoch_sr.py, or
#                                          # build_paft_cache.py — those scripts skip
#                                          # existing files by default, so a code fix
#                                          # has NO effect on your data until you
#                                          # either delete stale files or pass --force.
# ============================================================================

set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_CLEAR=0
SKIP_LORA=0
SKIP_CACHE=0
SKIP_LLAMA=0
FORCE=0

for arg in "$@"; do
    case $arg in
        --skip-clear)  SKIP_CLEAR=1 ;;
        --skip-lora)   SKIP_LORA=1  ;;
        --skip-cache)  SKIP_CACHE=1 ;;
        --skip-llama)  SKIP_LLAMA=1 ;;
        --plots-only)  SKIP_CLEAR=1; SKIP_LORA=1; SKIP_CACHE=1 ;;
        --force)       FORCE=1 ;;
    esac
done

FORCE_FLAG=""
[ "$FORCE" = "1" ] && FORCE_FLAG="--force"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives in scripts/ — project root is one level up
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

RESULTS_DIR="results/glue"
LLAMA_RESULTS_DIR="results/llama"
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
[ "$FORCE" = "1" ] && log "Mode: FORCE (recomputing even where output already exists)"
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

# ── Step 2: LoRA weight recovery — final checkpoint AND per-epoch series ──────
sep
if [ "$SKIP_LORA" = "0" ]; then
    log "STEP 2a: Recovering LoRA merged weights (final checkpoint)"
    log "  Merges adapter weights into base model, extracts per-head W_eff."
    log "  ~15 min. Rerun with --skip-lora once complete."

    # Remove stale merged files from previous runs (final/ AND epoch_*/ — this
    # find is filename-based, not path-based, so it catches both).
    STALE=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" -o -name "geometric_health_merged.pt" | wc -l)
    if [ "$STALE" -gt 0 ]; then
        log "  Removing $STALE stale merged files (final + per-epoch)"
        find "$RESULTS_DIR" -name "adapted_weights_merged.pt" -delete
        find "$RESULTS_DIR" -name "geometric_health_merged.pt" -delete
    fi

    if python3 -m analysis.recover_lora_weight --results_dir results $FORCE_FLAG 2>&1; then
        RECOVERED=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
        ok "LoRA recovery complete — $RECOVERED merged files created"
    else
        warn "LoRA recovery had errors — LoRA geometric data may be incomplete"
        RECOVERED=$(find "$RESULTS_DIR" -name "adapted_weights_merged.pt" | wc -l)
        log "  Merged files found: $RECOVERED"
    fi

    # STEP 2b was previously missing entirely from this pipeline: without it,
    # plot_training_dynamics.py and any per-epoch LoRA analysis silently run
    # on whatever (possibly stale, possibly absent) per-epoch merged health
    # files happen to already be on disk, since Step 2a above only writes to
    # final/, never epoch_*/. This is what actually produces the per-epoch
    # geometric_health_merged.pt files, and what patch_metrics_cache.py needs
    # downstream.
    log "STEP 2b: Computing per-epoch LoRA sr(W_eff) (training dynamics)"
    run_py analysis.compute_lora_epoch_sr $FORCE_FLAG

    log "STEP 2c: Patching metrics_cache.json with corrected per-epoch LoRA/PoLAR values"
    log "  (must run AFTER build_cache.py below, moved to end of Step 3)"
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

    # Must run after build_cache.py (needs metrics_cache.json to exist) and
    # after Step 2b (needs epoch_*/geometric_health_merged.pt to exist).
    if [ "$SKIP_LORA" = "0" ]; then
        log "STEP 3b: Patching per-epoch LoRA/PoLAR values into metrics_cache.json"
        run_py analysis.patch_metrics_cache
    fi
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

# Decay-law fits run first — plot_decay_law*.py (Step 6) depend on the
# decay_law_results*.json these produce.
run_py analysis.fit_decay_law
run_py analysis.fit_decay_law_wO
run_py analysis.tables.table_syntax_semantics
run_py analysis.tables.table_stable_rank
run_py analysis.tables.table_all_metrics
run_py analysis.tables.table_wO_metrics
run_py analysis.tables.table_correlation

# ── Step 6: Figures ───────────────────────────────────────────────────────────
sep
log "STEP 6: Generating figures → $FIGURES_DIR/"
# plot_efficiency_curve removed — deleted from the repo.

run_py analysis.plotting.plot_sr_scatter
run_py analysis.plotting.plot_training_dynamics
run_py analysis.plotting.plot_layer_profiles_delta
run_py analysis.plotting.plot_rotation_drift
run_py analysis.plotting.plot_collapse
run_py analysis.plotting.plot_eigenvalue_shift
run_py analysis.plotting.plot_geometric_heatmaps
run_py analysis.plotting.plot_decay_law
run_py analysis.plotting.plot_decay_law_wO

# ── Step 7: Dump results ──────────────────────────────────────────────────────
sep
log "STEP 7: Running dump_results → $DUMP_TXT"

python3 -m analysis.dump_results \
    --cache "$ANALYSIS_DIR/metrics_cache.json" \
    --paft  "$ANALYSIS_DIR/paft_cache.json" \
    --checkpoint_dir "$RESULTS_DIR" \
    2>&1 | tee "$DUMP_TXT"

ok "dump_results saved to $DUMP_TXT"

# ── Step 8: LLaMA-3.2-3B pipeline ────────────────────────────────────────────
# Independent of everything above — separate results directory, separate
# checkpoints, separate cache-building path (there is no build_cache.py
# equivalent for LLaMA; generate_paper_outputs.py builds its geometric
# tables directly from paft_cache-style data plus collect_llama_results.sh's
# accuracy summary).
sep
if [ "$SKIP_LLAMA" = "0" ]; then
    if [ -d "$LLAMA_RESULTS_DIR" ]; then
        log "STEP 8a: Recovering LLaMA LoRA merged weights"
        log "  Loads NF4 base + adapter, merges, extracts per-head W_V."
        log "  NOTE: W_O is not extracted here (VRAM-constrained scope — see"
        log "  Limitations). Only W_V geometric health is available for LLaMA."

        if python3 -m analysis.recover_lora_weights_llama \
            --results_dir "$LLAMA_RESULTS_DIR" $FORCE_FLAG 2>&1; then
            ok "LLaMA LoRA recovery complete"
        else
            warn "LLaMA LoRA recovery had errors — LLaMA geometric data may be incomplete"
        fi

        log "STEP 8b: Collecting LLaMA task-accuracy results"
        if bash scripts/collect_llama_results.sh \
            --results_dir "$LLAMA_RESULTS_DIR" --output_dir "$ANALYSIS_DIR" 2>&1; then
            ok "LLaMA accuracy results collected"
        else
            warn "collect_llama_results.sh had errors"
        fi

        log "STEP 8c: Generating LLaMA geometric tables, Q-drift, asymmetry, analysis dump"
        run_py analysis.generate_paper_outputs

        ok "LLaMA pipeline complete"
    else
        warn "STEP 8: $LLAMA_RESULTS_DIR not found — skipping LLaMA pipeline "
        warn "  (this is fine if you only have GLUE/DeBERTa results so far)"
    fi
else
    log "STEP 8: Skipped (--skip-llama)"
fi

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
if [ "$SKIP_LLAMA" = "0" ] && [ -d "$LLAMA_RESULTS_DIR" ]; then
    log "  LLaMA accuracy: $ANALYSIS_DIR/llama_results_summary.txt, llama_results.json (table_llama_performance.tex written by Step 8c)"
    log "  LLaMA geometric/Q-drift/analysis dump: see analysis.generate_paper_outputs output paths above"
fi
log ""

if [ -d "$FIGURES_DIR" ]; then
    NFIGS=$(find "$FIGURES_DIR" -name "*.pdf" | wc -l)
    log "  Figures ($NFIGS):"
    find "$FIGURES_DIR" -name "*.pdf" | sort | while read -r f; do
        echo "      $f"
    done
fi
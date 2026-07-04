# results/analysis — PAFT Paper Analysis Outputs

All files in this directory are auto-generated. Do not edit manually.
To regenerate everything run:

```bash
python3 analysis/generate_paper_outputs.py
```

To regenerate figures individually run the scripts in `analysis/plotting/`.

---

## Data Files

| File | Description |
|------|-------------|
| `metrics_cache.json` | Primary cache: per-task per-method spectral metrics for all GLUE experiments. Includes sr(W_eff), sr(ΔW), spectral entropy, effective rank, condition number, per-epoch sr, and per-layer sr. Patched with correct merged LoRA epoch sr values by `scripts/patch_metrics_cache.py`. |
| `paft_cache.json` | PAFT-specific cache: Q drift, S asymmetry ratios, and eigenvalue shift magnitudes per layer per task for all four PAFT variants. |
| `llama_results.json` | LLaMA-3.2-3B commonsense reasoning results: accuracy per method per task for BoolQ, HellaSwag, and ARC-Challenge. |
| `analysis_dump.txt` | Human-readable summary of all key numbers: GLUE scores, stable rank table, LLaMA results, and key paper statistics. |

---

## LaTeX Tables

### Main Paper

| File | Paper location | Description |
|------|---------------|-------------|
| `table_glue_performance.tex` | Section 6.1 | GLUE benchmark results for all 11 methods across 8 tasks with bold/underline formatting. |
| `table_llama_performance.tex` | Section 6.2 | LLaMA-3.2-3B commonsense accuracy for 5 methods across 3 tasks. |
| `table_stable_rank.tex` | Section 6.3 | Task-averaged sr(W_eff) before and after fine-tuning for all methods. |
| `table_q_drift.tex` | Section 6.7 | Frobenius drift of Q for all PAFT variants — all values 0.00e+00. |
| `table_asymmetry.tex` | Section 6.8 | S asymmetry ratio per task ordered by training set size; reveals monotone scaling with task complexity. |
| `table_llama_geometric.tex` | Section 6.9 | sr and condition number of W_V for LLaMA methods on BoolQ and HellaSwag. |

### Appendix

| File | Description |
|------|-------------|
| `table_all_metrics.tex` | Five spectral metrics (Δsr, ΔEntropy, ΔER, CondNum, Isotropy) task-averaged for all methods. |
| `table_sr_per_task.tex` | Final sr(W_eff) for every method × task combination. |
| `table_sr_delta_w.tex` | sr(ΔW_V) per task for all methods; high sr(ΔW) with low sr(W_eff) confirms Proposition 1. |
| `table_training_dynamics.tex` | sr(W_eff) per epoch on SST-2 for all methods including LoRA. |
| `table_per_layer_cola.tex` | Final sr(W_eff) at each of the 12 encoder layers for CoLA. |
| `table_per_layer_mrpc.tex` | Final sr(W_eff) at each of the 12 encoder layers for MRPC. |
| `table_llama_geometric_appendix.tex` | Full LLaMA geometric health table including ARC-Challenge. |

---

## Figures

### Main Paper

| File | Paper location | Description |
|------|---------------|-------------|
| `figures/efficiency_curve.pdf` | Section 6.1 | Pareto frontier of mean GLUE score vs trainable parameter count on log scale. |
| `figures/training_dynamics.pdf` | Section 6.5 | sr(W_eff) per epoch on SST-2; PAFT variants plateau while LoRA and PoLAR degrade. |
| `figures/sr_scatter.pdf` | Section 6.4 | sr(ΔW) vs sr(W_eff) scatter per method per task; empirical confirmation of Proposition 1. |
| `figures/rotation_drift.pdf` | Section 6.7 | Q Frobenius drift per layer for all PAFT variants; visual proof of exact Q invariance. |

### Appendix

| File | Description |
|------|-------------|
| `figures/collapse.pdf` | sr(W_eff) vs gradient steps on log scale; reveals scale-dependent geometric decay law. |
| `figures/layer_profiles_delta.pdf` | Δsr(W_eff) per encoder layer for CoLA, MRPC, QQP; shows Layer 0-1 damage and Layer 9 expansion. |
| `figures/eigenvalue_shift.pdf` | Mean eigenvalue shift magnitude and S asymmetry ratio per layer for PAFT variants. |
| `figures/geometric_heatmaps.pdf` | Layer × method heatmap of Δsr(W_eff) for STS-B; overview of geometric damage patterns. |

---

## How Tables Are Generated

All tables are produced by `analysis/generate_paper_outputs.py` which reads
from `metrics_cache.json`, `paft_cache.json`, and `llama_results.json`.
Run `python3 analysis/generate_paper_outputs.py` to regenerate all tables.

## How Figures Are Generated

Each figure has its own script in `analysis/plotting/`:

```bash
python3 analysis/plotting/plot_efficiency_curve.py
python3 analysis/plotting/plot_training_dynamics.py
python3 analysis/plotting/plot_sr_scatter.py
python3 analysis/plotting/plot_rotation_drift.py
python3 analysis/plotting/plot_collapse.py
python3 analysis/plotting/plot_layer_profiles_delta.py
python3 analysis/plotting/plot_eigenvalue_shift.py
python3 analysis/plotting/plot_geometric_heatmaps.py
```

## Data Pipeline
Results/glue/{task}/{method}/           Raw experiment outputs
├── metrics.json                     Task score + trainable params
├── init/geometric_health.pt         Pretrained W_eff geometry
├── epoch_N/geometric_health.pt      Per-epoch geometry (unmerged)
├── epoch_N/geometric_health_merged.pt  Per-epoch geometry (merged W_eff)
└── final/geometric_health_merged.pt Final merged W_eff geometry
results/llama/{task}/{method}/          LLaMA experiment outputs
├── init/geometric_health.pt
└── final/geometric_health.pt
scripts/compute_lora_epoch_sr.py        Computes merged LoRA epoch sr
scripts/patch_metrics_cache.py          Patches metrics_cache with correct values
analysis/generate_paper_outputs.py      Generates all LaTeX tables
analysis/plotting/plot_*.py             Generates all figures

## Notes

- LoRA epoch sr values were recomputed by merging HF adapter checkpoints
  with the base model using `scripts/compute_lora_epoch_sr.py`. The
  original `geometric_health.pt` files for LoRA contain unmerged base
  weights and should not be used for sr(W_eff) analysis.
- LLaMA PAFT is applied to W_V only due to VRAM constraints. LoRA targets
  both W_V and W_O. All geometric analyses compare W_V consistently.
- The HellaSwag pure_paft condition number (143.9) is a genuine finding:
  one epoch of eigenvalue-only adaptation on 39,905 examples drives
  sigma_min toward zero in several layers.
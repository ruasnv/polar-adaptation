geometric_health.py (526 lines) — the main paper figure
Reads geometric_health.pt from every epoch_*/ directory. Computes relative change from the pretrained baseline (init/geometric_health.pt) for all 6 metrics. Produces:

One heatmap per metric per projection (method × layer grid, red=degraded, blue=preserved)
A preservation score bar chart (single number per method per domain)
summary_table.csv for the paper table

powershellpython analysis/geometric_health.py --model gpt2_small
efficiency_curve.py (414 lines) — the Pareto plot
Reads only epoch_*/metrics.json — the lightest possible analysis (no tensors). Finds best eval_loss across all epochs per run. Parameter counts are hardcoded from the method definitions (verified against your training logs). Produces:

Per-domain Pareto scatter (log-scale x-axis, PAFT marked with ★)
Combined 3-panel figure for the paper
efficiency_table.csv

The Pareto frontier algorithm: a method is on the frontier if no other method achieves lower loss with fewer parameters. If hybrid_paft sits on or near the frontier alongside LoRA, that's your central result.
powershellpython analysis/efficiency_curve.py --model gpt2_small
geometric_audit.py (491 lines) — PAFT-specific rotation and eigenvalue analysis
Reads init/decomp_init.pt and epoch_*/paft_snapshot.pt. The first thing it does is check whether rotation drift is zero — max ||Q_t - Q_0||_F across all runs. If it's above 1e-4, Q wasn't frozen correctly and the training is invalid. If it's near zero, the core PAFT invariant holds. Then produces layer×epoch heatmaps of eigenvalue shift (where and when S changes), and eigenvalue trajectory line plots (which scaling directions the model learns to emphasise).
powershellpython analysis/geometric_audit.py --model gpt2_small
You can run any of these on partial data — if only news is done, it analyses news and skips domains with no sentinel file.

_utils.py — shared module imported by every script. Handles model loading for all 11 method types. The key function is load_trained_model(run_dir, cfg, device) which rebuilds the method, runs the polar decomposition, then loads the trained state_dict on top. get_hf_model(method) extracts the underlying GPT2LMHeadModel whether you have PAFT, SVF, LoRA or plain weights.
domain_metrics.py (Axis 1) — runs each model on task-specific test sets. News=ROUGE, legal/biomedical=accuracy+F1, code=BLEU. Saves final/domain_metrics.json per run. Run after each domain completes.
base_metrics.py (Axis 2) — uses lm-eval to run HellaSwag, ARC-Easy, LAMBADA, WikiText on every trained model. Computes the forgetting score as geometric mean of per-benchmark retention. The pretrained baseline is evaluated once and cached to disk.
residual_stability.py — loads final/adapted_weights.pt (no model reconstruction needed) and computes per-layer nuclear norm. Requires no GPU — pure tensor analysis.
layer_profiles.py — loads epoch_*/paft_snapshot.pt for PAFT methods and plots ||S_t - S_0||_F per layer per epoch. Shows where adaptation concentrates spatially.
domain_correlation.py — needs frozen to be complete for all domains (uses its eval_loss as the domain shift proxy), then scatter plots preservation score vs domain shift perplexity per method.
eigenvalue_semantics.py — projects vocabulary embeddings onto the dominant eigendirections of S_final - S_init. Finds which tokens semantically align with each learned direction. Most interpretable output of the whole pipeline.
dial_ablation.py — analyses the 15 dial ablation runs (Phase 3). Produces the task/geometry trade-off Pareto plot. Run only after Phase 3 completes.
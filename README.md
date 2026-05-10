# PAFT — Polar decomposition Attention Fine-Tuning

A geometric framework for parameter-efficient fine-tuning of transformers. PAFT decomposes OV circuit weight matrices via polar decomposition into rotation (Q) and scaling (S) components, then fine-tunes only the scaling matrices while freezing rotations.

## Structure

```
configs/        All experiment configuration — no hardcoding anywhere
paft/           Core library (importable package)
analysis/       Post-training analyses — strictly read-only
scripts/        Entry points — thin wrappers with no logic
results/        All outputs — gitignored
tests/          Unit tests for core math — run before any experiment
notebooks/      Exploration only — never used in the pipeline
```

## Quick Start

```bash
pip install -e ".[dev,analysis]"



## Methods

| Method           | Tunes                   | Params (GPT-2 small) |
|------------------|-------------------------|----------------------|
| frozen           | nothing                 | 0                    |
| bitfit           | biases only             | ~100K                |
| lora_r8          | low-rank ΔW             | ~300K                |
| lora_r64         | low-rank ΔW             | ~2.4M                |
| pure_paft        | λ of S_V, S_O           | ~18K                 |
| hybrid_paft      | full S_V, S_O           | ~1.2M                |
| safe_pure_paft   | λ + all biases          | ~120K                |
| safe_hybrid_paft | full S + all biases     | ~1.3M                |
| full_finetune    | all weights             | ~117M                |

## Critical Constraint

Each model × domain × method combination is trained **once**. All tensors needed for post-training analyses are saved during training via `paft/checkpointing/schema.py`. Do not re-run training to recover missing tensors — fix the schema and re-run from scratch.

## Tests

Run `pytest tests/` before any experiment. Key gates:

- `test_polar.py` — verifies `W ≈ Q @ S` and `Q^T Q ≈ I`
- `test_parameter_groups.py` — verifies exactly the right params have `requires_grad=True` per method
- `test_saver.py` — verifies all schema tensors are present after a dummy 10-step run
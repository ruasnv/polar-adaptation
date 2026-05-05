from paft.utils.config import get_config

# Load the flagship variant
cfg = get_config("gpt2_small", "news", "safe_hybrid_paft")
print(f"Method Name: {cfg['method']['name']}")
print(f"Tune Biases: {cfg['method']['tune']['biases']}")

# Updated keys for your RTX 5070 VRAM safety
print(f"Micro-batch: {cfg['training']['micro_batch_size']}")
print(f"Accumulation: {cfg['training']['gradient_accumulation_steps']}")

# Verify total effective batch size (should be 32)
total_batch = cfg['training']['micro_batch_size'] * cfg['training']['gradient_accumulation_steps']
print(f"Effective Batch Size: {total_batch}")
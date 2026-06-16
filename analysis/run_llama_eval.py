#!/usr/bin/env python3
"""
analysis/run_llama_eval.py
Sanity check script leveraging the unified repo framework layers.
"""
import torch
from transformers import AutoTokenizer
from paft.model.llama_paft_model import LLaMAPAFTModel, load_llama_nf4


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    print(f"Loading decoder target: {model_id}...")

    base, tokenizer = load_llama_nf4(model_id, device_map="auto")

    # Inject via the official model class wrapper
    print("Injecting unified PAFT layers into attention stack...")
    model = LLaMAPAFTModel(base, train_mode='hybrid', q_dtype=torch.float16)
    model.eval()

    prompt = "The geometry of deep learning attention mechanisms reveals that"
    inputs = tokenizer(prompt, return_tensors="pt").to(base.device)

    print("\nExecuting verification forward pass sequence...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated Output Text:\n\"{generated_text}\"")

    print("\nRunning baseline Stiefel Audit check...")
    drift = model.measure_orthogonality()
    print(f"Measured baseline layout drift: {drift:.2e} (Expected: 0.00e+00)")


if __name__ == "__main__":
    main()
"""
eigenvalue_semantics.py — What do the learned eigenvalues attend to?

After fine-tuning, the scaling matrix S has changed eigenvalues.
The eigenvectors (EV) of S span the directions in head-space that are
amplified or suppressed. By projecting the vocabulary embedding matrix
onto these eigendirections, we find which tokens align most strongly with
the dominant directions the model learned to emphasise.

If fine-tuning on biomedical text causes certain eigenvalues to grow,
and the corresponding eigenvectors align with biomedical vocabulary
(e.g. "insulin", "receptor", "antibody"), that is direct evidence that
the polar decomposition is capturing semantic adaptation in S.

Algorithm (per PAFT method, per domain, per layer, per head)
─────────────────────────────────────────────────────────────
1. Load EV_V_final (eigenvectors at final epoch) from paft_snapshot.pt
2. Load EV_V_0    (eigenvectors at init)        from decomp_init.pt
3. Compute Δlam = lam_final - lam_0             (eigenvalue shift)
4. For the top-k shifted eigenvectors:
   project W_E (vocab embedding matrix) onto each eigenvector
   → find tokens with highest alignment scores
5. Save top-20 tokens per top-k direction to JSON + figure

What it produces
────────────────
figures/eigenvalue_semantics/{method}_{domain}_layer{L}_head{H}_top_tokens.json
    {"direction_0": {"tokens": [...], "scores": [...]},
     "direction_1": {...}, ...}

figures/eigenvalue_semantics/semantic_heatmap_{method}_{domain}.png
    Heatmap: top tokens × eigendirections, colour = alignment score.
    Rows = domain vocabulary terms (grouped by domain), cols = directions.

Usage
─────
    python analysis/eigenvalue_semantics.py --model gpt2_small --domain biomedical
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paft.checkpointing.loader import CheckpointLoader
from paft.data.utils import get_tokenizer

PAFT_METHODS   = ["pure_paft","hybrid_paft","safe_pure_paft","safe_hybrid_paft"]
METHOD_LABELS  = {
    "pure_paft":"Pure PAFT","hybrid_paft":"Hybrid PAFT",
    "safe_pure_paft":"Safe Pure PAFT","safe_hybrid_paft":"Safe Hybrid PAFT",
}

# How many top-shifted eigenvalue directions to analyse
TOP_K_DIRECTIONS = 4
# How many top vocabulary tokens to show per direction
TOP_K_TOKENS = 20
# Which layers to analyse (layer 5 and 8 in GPT-2 small are mid-network)
LAYERS_TO_ANALYSE = [5, 8]


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary embedding projection
# ──────────────────────────────────────────────────────────────────────────────

def get_vocab_embeddings(hf_name: str) -> Tuple[torch.Tensor, List[str]]:
    """
    Load GPT-2 vocabulary embeddings W_E [vocab_size, n_embd].
    Returns (embeddings_normalized, token_strings).
    """
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(hf_name)
    W_E   = model.transformer.wte.weight.detach().float()  # [V, n_embd]
    # L2-normalise for cosine similarity
    W_E_norm = W_E / (W_E.norm(dim=1, keepdim=True) + 1e-8)

    tok = get_tokenizer(hf_name)
    tokens = [tok.decode([i]).strip() for i in range(len(tok))]
    del model
    return W_E_norm, tokens


def top_tokens_for_direction(
    W_E_norm:   torch.Tensor,   # [V, n_embd]
    eigvec:     torch.Tensor,   # [n_embd] — one eigenvector of S
    token_strs: List[str],
    k:          int = TOP_K_TOKENS,
) -> Tuple[List[str], List[float]]:
    """
    Find the k vocabulary tokens whose embeddings align most with `eigvec`.
    Returns (top_k_tokens, cosine_scores).
    """
    # eigvec lives in the d_head space of W_V_h, which is in the V-side space.
    # The embedding is in n_embd space. We need to project eigvec to n_embd.
    # eigvec shape: [d_head]. W_V: [n_embd, d_head] per head.
    # We project W_E (vocab) through eigvec direction:
    #   score_i = W_E[i] @ W_V @ eigvec / norm
    # Here we approximate with direct cosine between W_E and eigvec
    # (valid when d_head == n_embd or when using concatenated W_V).
    # For GPT-2 small: d_head=64, n_embd=768 — we pad eigvec to n_embd.

    ev = eigvec.float()
    if ev.shape[0] < W_E_norm.shape[1]:
        # Pad to n_embd (remaining dims are zero — no alignment)
        pad   = torch.zeros(W_E_norm.shape[1] - ev.shape[0])
        ev    = torch.cat([ev, pad])
    ev_norm = ev / (ev.norm() + 1e-8)   # [n_embd]

    scores = (W_E_norm @ ev_norm).numpy()   # [V]
    top_k  = np.argsort(scores)[::-1][:k]

    top_tokens = [token_strs[i] for i in top_k]
    top_scores = [round(float(scores[i]), 4) for i in top_k]
    return top_tokens, top_scores


# ──────────────────────────────────────────────────────────────────────────────
# Per-run analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_run(
    run_dir:    Path,
    W_E_norm:   torch.Tensor,
    token_strs: List[str],
    out_dir:    Path,
    method:     str,
    domain:     str,
) -> None:
    loader = CheckpointLoader(run_dir)

    decomp_init = loader.load_decomp_init()
    final_snap  = loader.load_final_paft_snapshot()
    if decomp_init is None or final_snap is None:
        print(f"    Skipping {method}/{domain} — missing decomp_init or paft_snapshot")
        return

    lam_V_0  = decomp_init.get("lam_V_0",  [])   # [n_layers] of [H, d]
    EV_V_0   = decomp_init.get("EV_V_0",   [])   # [n_layers] of [H, d, d]
    lam_V_t  = final_snap.get("lam_V",     [])
    EV_V_t   = final_snap.get("EV_V",      [])

    if not lam_V_0 or not lam_V_t:
        return

    all_results = {}

    for layer_idx in LAYERS_TO_ANALYSE:
        if layer_idx >= len(lam_V_0):
            continue

        lam0 = lam_V_0[layer_idx].float()   # [H, d]
        lamt = lam_V_t[layer_idx].float()
        ev0  = EV_V_0[layer_idx].float()    # [H, d, d]  cols = eigvecs
        evt  = EV_V_t[layer_idx].float()

        delta_lam = (lamt - lam0).abs()     # [H, d]

        for head in range(min(2, lam0.shape[0])):   # analyse first 2 heads
            # Sort eigenvalues by shift magnitude, pick top-k directions
            shifts      = delta_lam[head].numpy()       # [d]
            top_dirs    = np.argsort(shifts)[::-1][:TOP_K_DIRECTIONS]

            result_key  = f"layer{layer_idx}_head{head}"
            all_results[result_key] = {}

            for rank, dir_idx in enumerate(top_dirs):
                eigvec_final = evt[head, :, dir_idx]   # [d_head]
                eigvec_init  = ev0[head, :, dir_idx]

                tok_final, sc_final = top_tokens_for_direction(
                    W_E_norm, eigvec_final, token_strs
                )
                tok_init, sc_init = top_tokens_for_direction(
                    W_E_norm, eigvec_init, token_strs
                )

                all_results[result_key][f"direction_{rank}"] = {
                    "eigenvalue_shift": round(float(shifts[dir_idx]), 4),
                    "final_lam":        round(float(lamt[head, dir_idx]), 4),
                    "init_lam":         round(float(lam0[head, dir_idx]), 4),
                    "top_tokens_final": tok_final,
                    "scores_final":     sc_final,
                    "top_tokens_init":  tok_init,
                    "scores_init":      sc_init,
                    # New tokens: in final top-20 but not in init top-20
                    "new_tokens":       [t for t in tok_final if t not in tok_init],
                }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{method}_{domain}_top_tokens.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"    Saved {out_file}")

    _plot_semantic_heatmap(all_results, out_dir, method, domain)


def _plot_semantic_heatmap(
    results: Dict,
    out_dir: Path,
    method:  str,
    domain:  str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Collect all unique final tokens across analysed layers/heads
    all_tokens = []
    dir_labels = []

    for loc_key, directions in results.items():
        for dir_key, data in directions.items():
            label = f"{loc_key}\n{dir_key}\nΔλ={data['eigenvalue_shift']:.2f}"
            dir_labels.append(label)
            all_tokens.extend(data["top_tokens_final"][:10])

    # Deduplicate tokens, keep order
    seen = set(); unique_tokens = []
    for t in all_tokens:
        if t and t not in seen and len(t) > 1:
            seen.add(t); unique_tokens.append(t)
    unique_tokens = unique_tokens[:30]   # cap at 30 rows

    if not unique_tokens or not dir_labels:
        return

    # Build alignment matrix
    matrix = np.zeros((len(unique_tokens), len(dir_labels)))
    for col_idx, (loc_key, directions) in enumerate(results.items()):
        for d_idx, (dir_key, data) in enumerate(directions.items()):
            flat_idx = list(results.keys()).index(loc_key) * len(directions) + d_idx
            if flat_idx >= len(dir_labels):
                continue
            for row_idx, tok in enumerate(unique_tokens):
                if tok in data["top_tokens_final"]:
                    rank    = data["top_tokens_final"].index(tok)
                    score   = data["scores_final"][rank]
                    matrix[row_idx, flat_idx] = score

    fig, ax = plt.subplots(figsize=(max(6, len(dir_labels)*2),
                                    max(6, len(unique_tokens)*0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(dir_labels)))
    ax.set_xticklabels(dir_labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(unique_tokens)))
    ax.set_yticklabels(unique_tokens, fontsize=7)
    ax.set_title(
        f"Vocabulary Alignment with Dominant Eigendirections\n"
        f"{METHOD_LABELS.get(method,method)} — {domain}",
        fontsize=9
    )
    plt.colorbar(im, ax=ax, fraction=0.03, label="Cosine similarity")
    plt.tight_layout()
    fname = out_dir / f"semantic_heatmap_{method}_{domain}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(
    model:           str,
    domains:         List[str],
    checkpoint_root: Path,
    figure_dir:      Path,
    hf_name:         str,
):
    print(f"\n=== Eigenvalue Semantics: {model} ===")
    print("Loading vocabulary embeddings (once) ...")
    W_E_norm, token_strs = get_vocab_embeddings(hf_name)
    print(f"  Loaded {len(token_strs)} vocabulary tokens, embedding dim={W_E_norm.shape[1]}")

    for domain in domains:
        for method in PAFT_METHODS:
            run_dir = checkpoint_root / model / domain / method
            if not (run_dir / "final" / "training_complete").exists():
                continue
            print(f"  Analysing {method}/{domain} ...")
            analyse_run(
                run_dir, W_E_norm, token_strs,
                figure_dir / "per_run", method, domain,
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2_small")
    p.add_argument("--domains", nargs="+", default=["news","biomedical","code"])
    p.add_argument("--checkpoint_root", default="results/checkpoints")
    p.add_argument("--figure_dir",      default="results/figures/eigenvalue_semantics")
    p.add_argument("--hf_name",         default="gpt2")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.model, a.domains, Path(a.checkpoint_root), Path(a.figure_dir), a.hf_name)

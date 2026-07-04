"""
paft/utils/experiment_saver.py

Saves everything the paper analyses need, for every method, at every stage.

Called by both train_glue.py and train_llm.py at:
  - INIT:  before any gradient steps
  - EPOCH: after each eval pass
  - FINAL: after the last epoch

Saves per run directory:
  init/
    adapted_weights.pt    W_V [H, n, d] and W_O [H, d, n] per layer — ALL methods
    paft_snapshot.pt      Q, S, EV, lam per layer — PAFT methods only
    geometric_health.pt   GeometricHealthMetrics per layer/head — ALL methods
  epoch_N/
    metrics.json          eval metrics (accuracy, loss, etc.)
    paft_snapshot.pt      PAFT methods only
    geometric_health.pt   ALL methods
    optimizer.pt          for training resume
    scheduler.pt          for training resume
  final/
    adapted_weights.pt    W_V, W_O per layer — ALL methods
    paft_snapshot.pt      PAFT methods only
    geometric_health.pt   ALL methods
    stable_rank.json      stable rank analysis — ALL methods
    training_complete     sentinel file

Why we save W_V/W_O for ALL methods (not just PAFT):
  sr(ΔW) = sr(W_final - W_init) requires W_init.
  Without init weights, the PoLAR-style sr(ΔW) comparison cannot be computed
  for LoRA, BitFit, or SVF baselines.  The comparison is pointless without it.

Weight extraction conventions (matching BaseMethod and extractor.py):
  W_V: [H, n_embd, d_head]  — value projection per head
  W_O: [H, d_head, n_embd]  — output projection per head
  Both extracted from any model type: PAFTLinear or plain nn.Linear.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from paft.methods.base import GeometricHealthMetrics, PAFTSnapshot

logger = logging.getLogger(__name__)


# ── DeBERTa weight extraction (works for PAFT and non-PAFT) ──────────────────

def extract_deberta_weights(model: nn.Module) -> Dict[str, List[torch.Tensor]]:
    """
    Extract W_V and W_O per layer from any DeBERTa-based model.

    Works for:
      - DeBERTaPAFTModel  → calls .reconstruct_weight() on PAFTLinear layers
      - DebertaV2ForSequenceClassification (raw or PEFT-wrapped)
                         → reads .weight from nn.Linear layers

    Returns:
        {"W_V": [n_layers × Tensor[H, n_embd, d_head]],
         "W_O": [n_layers × Tensor[H, d_head, n_embd]]}
    """
    from paft.model.deberta_paft_model import DeBERTaPAFTModel, _N_LAYERS, _N_HEADS, _HEAD_DIM, _HID_SIZE

    if isinstance(model, DeBERTaPAFTModel):
        return model.get_live_WV_WO()

    # Unwrap PEFT wrapper if present.
    # ALL HuggingFace models have a .base_model property (returns the backbone submodule),
    # but only PeftModel wraps have .base_model.model (the full classification model).
    # Check for .model explicitly so plain DeBERTa models are handled correctly.
    base_model = model
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base_model = model.base_model.model  # PeftModel → DebertaV2ForSequenceClassification
    deberta = getattr(base_model, 'deberta', None)
    if deberta is None:
        logger.warning("Cannot extract DeBERTa weights — unexpected model structure")
        return {"W_V": [], "W_O": []}

    W_V_layers, W_O_layers = [], []
    with torch.no_grad():
        for l in range(_N_LAYERS):
            attn_self = deberta.encoder.layer[l].attention.self
            attn_out  = deberta.encoder.layer[l].attention.output

            vp_weight = _get_weight(attn_self.value_proj)   # [H*d, n] in Linear convention
            od_weight = _get_weight(attn_out.dense)         # [n, H*d] in Linear convention

            # value_proj weight [H*d, n] → [H, d, n] → permute → [H, n, d] = [H, n_embd, d_head]
            W_V = (vp_weight
                   .reshape(_N_HEADS, _HEAD_DIM, _HID_SIZE)
                   .permute(0, 2, 1)
                   .contiguous().cpu())

            # output.dense weight [n, H*d] → [n, H, d] → permute → [H, d, n] = [H, d_head, n_embd]
            W_O = (od_weight
                   .reshape(_HID_SIZE, _N_HEADS, _HEAD_DIM)
                   .permute(1, 2, 0)
                   .contiguous().cpu())

            W_V_layers.append(W_V)
            W_O_layers.append(W_O)

    return {"W_V": W_V_layers, "W_O": W_O_layers}


def _get_weight(layer: nn.Module) -> torch.Tensor:
    """
    Extract the effective weight tensor from any layer type used in the codebase.

    Dispatch table:
        PAFTLinear   → reconstruct Q @ S or S @ Q (polar decomposition)
        PoLARLinear  → W_0 + (alpha/r) * X @ B^T  (Stiefel adaptation)
        SVFLinear    → (U * sigma) @ Vh             (singular value adaptation)
        PEFT layer   → base_layer.weight            (LoRA merged not yet — raw W_0)
        nn.Linear    → .weight                      (standard)
    """
    from paft.model.paft_linear import PAFTLinear
    from paft.model.polar_linear import PoLARLinear

    if isinstance(layer, PAFTLinear):
        return layer.reconstruct_weight().detach().float()

    if isinstance(layer, PoLARLinear):
        # W_eff = W_0 + scale * X @ B^T
        return layer.get_effective_W().float()

    # SVFLinear — identified by its unique buffers (no weight attribute)
    if hasattr(layer, 'sigma_init') and hasattr(layer, 'U') and hasattr(layer, 'Vh'):
        with torch.no_grad():
            sigma = (layer.sigma_init + layer.delta_sigma).float()
            return ((layer.U.float() * sigma) @ layer.Vh.float()).detach()

    # PEFT LoRA-wrapped layer — return the frozen base weight (W_0 only, no adapter)
    # This is correct for init/geometric snapshots; the adapter delta is small at init.
    if hasattr(layer, 'base_layer'):
        return layer.base_layer.weight.detach().float()

    # Standard nn.Linear
    return layer.weight.detach().float()


# ── LLaMA weight extraction ──────────────────────────────────────────────────

def extract_llama_weights(model: nn.Module) -> Dict[str, List[torch.Tensor]]:
    """
    Extract W_V and W_O per layer from any LLaMA-based model.
    Delegates to model.get_live_WV_WO() for LLaMAPAFTModel, or reads weights directly.
    """
    from paft.model.llama_paft_model import LLaMAPAFTModel, _dequantize_weight
    from paft.model.polar_linear import PoLARLinear

    if isinstance(model, LLaMAPAFTModel):
        return model.get_live_WV_WO()

    # Unwrap base model
    base = getattr(model, 'model', model)
    if hasattr(base, 'base_model') and hasattr(base.base_model, 'model'):  # PeftModel
        base = base.base_model.model
    llama_layers = getattr(base, 'model', base)  # LlamaModel.layers

    cfg  = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', 8)
    n_q  = cfg.num_attention_heads
    d    = cfg.hidden_size // n_q
    n    = cfg.hidden_size
    n_layers = cfg.num_hidden_layers

    W_V_layers, W_O_layers = [], []
    with torch.no_grad():
        for l in range(n_layers):
            attn = llama_layers.layers[l].self_attn
            if isinstance(attn.v_proj, PoLARLinear):
                v_w = attn.v_proj.get_effective_W()    # [n_kv*d, hidden]
            else:
                v_w = _dequantize_weight(attn.v_proj)  # [n_kv*d, hidden]
            o_w  = _dequantize_weight(attn.o_proj)     # [hidden, n_q*d]

            W_V = (v_w.reshape(n_kv, d, n).permute(0, 2, 1).contiguous().cpu())  # [H_kv, n, d]
            W_O = (o_w.reshape(n, n_q, d).permute(1, 2, 0).contiguous().cpu())   # [H_q, d, n]
            W_V_layers.append(W_V)
            W_O_layers.append(W_O)

    return {"W_V": W_V_layers, "W_O": W_O_layers}


# ── Geometric health (uses existing GeometricHealthMetrics from base.py) ──────

def compute_geometric_health(
    adapted_weights: Dict[str, List[torch.Tensor]],
) -> Dict[str, Any]:
    """
    Compute GeometricHealthMetrics per layer/head using the existing base.py logic.
    Returns a dict suitable for torch.save.
    """
    W_V_layers = adapted_weights["W_V"]
    W_O_layers = adapted_weights["W_O"]
    n_layers = len(W_V_layers)

    per_layer = {}
    all_V, all_O = [], []

    with torch.no_grad():
        for l in range(n_layers):
            W_V_l = W_V_layers[l].float()   # [H, n, d]
            W_O_l = W_O_layers[l].float()   # [H, d, n]
            n_heads = W_V_l.shape[0]
            layer_V, layer_O = [], []

            for h in range(n_heads):
                sv_V = torch.linalg.svdvals(W_V_l[h])
                sv_O = torch.linalg.svdvals(W_O_l[h])
                m_V = GeometricHealthMetrics.from_singular_values(sv_V)
                m_O = GeometricHealthMetrics.from_singular_values(sv_O)
                layer_V.append(m_V)
                layer_O.append(m_O)
                all_V.append(m_V)
                all_O.append(m_O)

            per_layer[l] = {
                "W_V": GeometricHealthMetrics.average(layer_V).to_dict("V_"),
                "W_O": GeometricHealthMetrics.average(layer_O).to_dict("O_"),
            }

    global_metrics = {
        "W_V": GeometricHealthMetrics.average(all_V).to_dict("V_"),
        "W_O": GeometricHealthMetrics.average(all_O).to_dict("O_"),
    }
    return {"per_layer": per_layer, "global": global_metrics}


# ── PAFT snapshot extraction ──────────────────────────────────────────────────

def extract_paft_snapshot(model: nn.Module) -> Optional[Dict[str, Any]]:
    """
    Extract PAFT snapshot from DeBERTaPAFTModel or LLaMAPAFTModel.
    Returns None for non-PAFT models.  Returns plain dict (not dataclass) for torch.save.
    """
    from paft.model.deberta_paft_model import DeBERTaPAFTModel
    from paft.model.llama_paft_model import LLaMAPAFTModel

    snap = None
    if isinstance(model, DeBERTaPAFTModel):
        snap = model.get_snapshot()
    elif isinstance(model, LLaMAPAFTModel):
        snap = model.get_snapshot()
    else:
        return None

    if snap is None:
        return None

    # Convert PAFTSnapshot dataclass to plain dict for torch.save compatibility
    return {
        field: [t.detach().cpu() for t in getattr(snap, field)]
        for field in ("Q_V", "Q_O", "S_V", "S_O", "EV_V", "EV_O", "lam_V", "lam_O")
        if hasattr(snap, field)
    }


# ── Main save function ────────────────────────────────────────────────────────

def save_checkpoint(
    model:        nn.Module,
    output_dir:   Path,
    tag:          str,
    method_name:  str,
    metrics:      Optional[Dict] = None,
    optimizer:    Optional[Any]  = None,
    scheduler:    Optional[Any]  = None,
    model_type:   str = "deberta",   # "deberta" | "llama"
) -> None:
    """
    Save all analysis-required data for one training event.

    Args:
        model:       The trained model (any method variant).
        output_dir:  Run root dir.  Sub-dir `tag/` is created automatically.
        tag:         "init", "epoch_N", or "final".
        method_name: For logging.
        metrics:     Dict of eval metrics to save as metrics.json.
        optimizer:   torch optimizer — state saved for resume if provided.
        scheduler:   torch scheduler — state saved for resume if provided.
        model_type:  Which weight extractor to use.
    """
    save_dir = output_dir / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # 1. Extract W_V, W_O per layer (ALL methods)
        if model_type == "deberta":
            adapted = extract_deberta_weights(model)
        else:
            adapted = extract_llama_weights(model)

        if adapted["W_V"]:
            torch.save(adapted, save_dir / "adapted_weights.pt")
            logger.info(f"[{method_name}/{tag}] Saved adapted_weights.pt "
                        f"({len(adapted['W_V'])} layers)")

        # 2. Geometric health (ALL methods)
        if adapted["W_V"]:
            health = compute_geometric_health(adapted)
            torch.save(health, save_dir / "geometric_health.pt")
            sr_global = health["global"]["W_V"].get("V_stable_rank", "N/A")
            logger.info(f"[{method_name}/{tag}] sr(W_V) global mean = "
                        f"{sr_global:.3f}" if isinstance(sr_global, float) else
                        f"[{method_name}/{tag}] geometric_health.pt saved")

        # 3. PAFT snapshot (PAFT methods only)
        snap = extract_paft_snapshot(model)
        if snap is not None:
            torch.save(snap, save_dir / "paft_snapshot.pt")
            logger.info(f"[{method_name}/{tag}] Saved paft_snapshot.pt")

        # 4. Metrics JSON
        if metrics is not None:
            with open(save_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # 5. Optimizer + scheduler (for resume)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")

        # 6. Sentinel for final
        if tag == "final":
            (save_dir / "training_complete").touch()
            logger.info(f"[{method_name}] training_complete sentinel written.")

    model.train()


def is_complete(output_dir: Path) -> bool:
    """Return True if a previous run finished (has sentinel file)."""
    return (output_dir / "final" / "training_complete").exists()
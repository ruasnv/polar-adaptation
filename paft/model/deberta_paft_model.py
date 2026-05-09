"""
DeBERTaPAFTModel — wraps DebertaV2ForSequenceClassification with PAFT.

Architecture:  DeBERTa-v3-base
  hidden_size  = 768
  num_heads    = 12
  head_dim     = 64
  num_layers   = 12

PAFT targets two projections per attention layer:
  value_proj  nn.Linear [768, 768]  — replaced with PAFTLinear (row mode)
  output.dense nn.Linear [768, 768] — replaced with PAFTLinear (col mode)

Q/K projections are left completely frozen — same scope as GPT-2 PAFT.
The classification head (pooler + classifier) is ALWAYS trainable —
without it GLUE tasks cannot be learned by any method.

Weight layout (DeBERTa uses standard nn.Linear, not Conv1D):
  value_proj.weight  [out=768, in=768]  — rows split per head: [h*64:(h+1)*64, :]
  output.dense.weight [out=768, in=768] — cols split per head: [:, h*64:(h+1)*64]

All decomposition runs on CPU before .to(device) call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, DebertaV2Config

from paft.model.paft_linear import PAFTLinear

logger = logging.getLogger(__name__)

# DeBERTa-v3-base architecture constants
_N_LAYERS  = 12
_N_HEADS   = 12
_HEAD_DIM  = 64
_HID_SIZE  = 768


@dataclass
class DeBERTaPAFTLayerSnapshot:
    """Decomposition state for one DeBERTa attention layer."""
    Q_V:   torch.Tensor   # [H, d, n]  frozen Q for value_proj (row mode)
    Q_O:   torch.Tensor   # [H, n, d]  frozen Q for output.dense (col mode)
    S_V:   torch.Tensor   # [H, d, d]  scaling for value_proj
    S_O:   torch.Tensor   # [H, d, d]  scaling for output.dense
    EV_V:  torch.Tensor   # [H, d, d]  eigenvectors of initial S_V
    EV_O:  torch.Tensor   # [H, d, d]  eigenvectors of initial S_O
    lam_V: torch.Tensor   # [H, d]     eigenvalues of initial S_V
    lam_O: torch.Tensor   # [H, d]     eigenvalues of initial S_O


@dataclass
class DeBERTaPAFTSnapshot:
    """Snapshot of all PAFT state across all layers. Mirrors PAFTSnapshot for GPT-2."""
    Q_V:   List[torch.Tensor] = field(default_factory=list)
    Q_O:   List[torch.Tensor] = field(default_factory=list)
    S_V:   List[torch.Tensor] = field(default_factory=list)
    S_O:   List[torch.Tensor] = field(default_factory=list)
    EV_V:  List[torch.Tensor] = field(default_factory=list)
    EV_O:  List[torch.Tensor] = field(default_factory=list)
    lam_V: List[torch.Tensor] = field(default_factory=list)
    lam_O: List[torch.Tensor] = field(default_factory=list)


class DeBERTaPAFTModel(nn.Module):
    """
    DeBERTa-v3 with value_proj and output.dense replaced by PAFTLinear.

    Usage:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=num_labels
        )
        model = DeBERTaPAFTModel(base_model, train_mode='hybrid')
        # Then method._configure_parameters() controls what is frozen/unfrozen.
    """

    def __init__(
        self,
        base_model,
        train_mode: str = 'hybrid',
        q_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.base       = base_model
        self.train_mode = train_mode
        self.q_dtype    = q_dtype
        self._n_layers  = _N_LAYERS

        logger.info(
            f"DeBERTaPAFTModel: decomposing {_N_LAYERS} layers × {_N_HEADS} heads "
            f"(train_mode={train_mode}) on CPU ..."
        )
        self._inject_paft_layers()
        logger.info("DeBERTaPAFTModel: decomposition complete.")

    # ── injection ─────────────────────────────────────────────────────────────

    def _inject_paft_layers(self) -> None:
        """Replace value_proj and output.dense with PAFTLinear in every layer."""
        for l in range(self._n_layers):
            attn_self   = self.base.deberta.encoder.layer[l].attention.self
            attn_output = self.base.deberta.encoder.layer[l].attention.output

            # ── value_proj [768, 768] — row mode ──────────────────────────────
            vp = attn_self.value_proj
            attn_self.value_proj = PAFTLinear(
                weight       = vp.weight.detach().float(),
                bias         = vp.bias.detach().float() if vp.bias is not None else None,
                n_heads      = _N_HEADS,
                head_dim     = _HEAD_DIM,
                decomp_mode  = 'row',
                train_mode   = self.train_mode,
                q_dtype      = self.q_dtype,
            )
            del vp

            # ── output.dense [768, 768] — col mode ────────────────────────────
            od = attn_output.dense
            attn_output.dense = PAFTLinear(
                weight       = od.weight.detach().float(),
                bias         = od.bias.detach().float() if od.bias is not None else None,
                n_heads      = _N_HEADS,
                head_dim     = _HEAD_DIM,
                decomp_mode  = 'col',
                train_mode   = self.train_mode,
                q_dtype      = self.q_dtype,
            )
            del od

            if (l + 1) % 4 == 0:
                logger.info(f"  {l + 1}/{self._n_layers} layers replaced")

    # ── mode switching ────────────────────────────────────────────────────────

    def set_train_mode(self, mode: str) -> None:
        """Switch all PAFTLinear layers between 'pure' and 'hybrid'."""
        if mode not in ('pure', 'hybrid'):
            raise ValueError(f"mode must be 'pure' or 'hybrid', got {mode!r}")
        self.train_mode = mode
        for _, vp, od in self._iter_paft_layers():
            vp.train_mode = mode
            od.train_mode = mode

    # ── iteration ─────────────────────────────────────────────────────────────

    def _iter_paft_layers(self) -> Iterator[Tuple[int, PAFTLinear, PAFTLinear]]:
        """Yield (layer_idx, value_proj_paft, output_dense_paft)."""
        for l in range(self._n_layers):
            vp = self.base.deberta.encoder.layer[l].attention.self.value_proj
            od = self.base.deberta.encoder.layer[l].attention.output.dense
            if not isinstance(vp, PAFTLinear) or not isinstance(od, PAFTLinear):
                raise RuntimeError(
                    f"Layer {l} projections were not replaced — "
                    "was _inject_paft_layers() called?"
                )
            yield l, vp, od

    # ── geometric accessors ───────────────────────────────────────────────────

    def get_snapshot(self) -> DeBERTaPAFTSnapshot:
        """Collect Q, S, EV, lam for all layers. Returns CPU tensors."""
        snap = DeBERTaPAFTSnapshot()
        with torch.no_grad():
            for _, vp, od in self._iter_paft_layers():
                Q_V, S_V = vp.get_Q_S()
                Q_O, S_O = od.get_Q_S()
                snap.Q_V.append(Q_V)
                snap.Q_O.append(Q_O)
                snap.S_V.append(S_V)
                snap.S_O.append(S_O)
                # EV and lam
                if self.train_mode == 'pure':
                    snap.EV_V.append(vp.EV.cpu())
                    snap.EV_O.append(od.EV.cpu())
                    snap.lam_V.append(vp.lam.detach().cpu())
                    snap.lam_O.append(od.lam.detach().cpu())
                else:
                    # Compute EV/lam from current S for hybrid (snapshot compatibility)
                    for S, ev_list, lam_list in [
                        (S_V, snap.EV_V, snap.lam_V),
                        (S_O, snap.EV_O, snap.lam_O),
                    ]:
                        ev_heads, lam_heads = [], []
                        for h in range(_N_HEADS):
                            S_sym = (S[h] + S[h].T) / 2.0
                            lam_h, ev_h = torch.linalg.eigh(S_sym)
                            ev_heads.append(ev_h.flip(1))
                            lam_heads.append(lam_h.flip(0))
                        ev_list.append(torch.stack(ev_heads))
                        lam_list.append(torch.stack(lam_heads))
        return snap

    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Return effective W_V and W_O per layer (all heads combined).
        W_V [H, n, d], W_O [H, d, n] — consistent with GPT-2 PAFT convention.
        """
        W_V_layers, W_O_layers = [], []
        with torch.no_grad():
            for _, vp, od in self._iter_paft_layers():
                # vp reconstructs [H*d, n] → reshape to [H, d, n]
                W_V = vp.reconstruct_weight().detach().reshape(_N_HEADS, _HEAD_DIM, _HID_SIZE)
                # od reconstructs [n, H*d] → reshape/permute to [H, n, d] ... wait
                # col mode: W_h = Q_h @ S_h, W [n_out, H*d]
                # Per head: W[:, h*d:(h+1)*d] = [n, d]
                W_O_full = od.reconstruct_weight().detach()   # [n, H*d]
                W_O = W_O_full.reshape(_HID_SIZE, _N_HEADS, _HEAD_DIM).permute(1, 0, 2)  # [H, n, d]
                W_V_layers.append(W_V.cpu())
                W_O_layers.append(W_O.cpu())
        return {"W_V": W_V_layers, "W_O": W_O_layers}

    def measure_orthogonality(self) -> Dict[str, float]:
        """
        Report mean ||Q_h^T Q_h - I||_F across all layers and heads.
        Used to validate that fp16 Q storage doesn't break orthogonality.
        """
        vp_errs, od_errs = [], []
        for _, vp, od in self._iter_paft_layers():
            vp_errs.append(vp.orthogonality_error())
            od_errs.append(od.orthogonality_error())
        return {
            "mean_ortho_error_V": sum(vp_errs) / len(vp_errs),
            "mean_ortho_error_O": sum(od_errs) / len(od_errs),
        }

    # ── forward / HuggingFace passthrough ─────────────────────────────────────

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    @property
    def config(self):
        return self.base.config

    def gradient_checkpointing_enable(self, **kwargs):
        self.base.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        self.base.gradient_checkpointing_disable(**kwargs)
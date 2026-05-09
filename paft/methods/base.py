"""
BaseMethod — the single interface the trainer touches for all 11 methods.

Design contracts enforced here:

  1. CPU offloading of initial decomposition
     build() always runs the decomposition pass on CPU, then moves the resulting
     buffers/parameters to the target device.  This keeps the VRAM footprint to
     zero during the expensive O(n_layers * n_heads) SVD/polar pass.  The full
     base model weights are loaded from HuggingFace in fp32, decomposed, and only
     the needed components are kept on device.

  2. Method-agnostic trainer loop
     The trainer calls exactly:
         loss = method.forward(input_ids, attention_mask, labels)
         loss.backward()
         method.pre_optimizer_step()     # no-op for all 11 methods
         optimizer.step()
         optimizer.zero_grad()
     Nothing else.  No method-specific branches in the trainer.

  3. Landing field via register_hook, not via pre_optimizer_step
     PoLAR registers gradient hooks on X and Y during build().  The hooks fire
     during backward() and modify .grad in-place before the optimizer sees it.
     pre_optimizer_step() remains a no-op even for PoLAR.  This keeps AdamW
     identical across all methods, making optimizer dynamics a non-variable.

  4. Geometric health computed on live weights, not stored state
     geometric_health_snapshot() calls get_live_WV_WO(), which each subclass
     implements.  For additive methods it returns W_0 + ΔW.  For surgery methods
     it reconstructs from stored components.  The health computation is then
     identical regardless of method internals.

  5. Gradient checkpointing location
     Surgery models (PAFTModel, SVFModel) enable gradient checkpointing internally
     on their transformer blocks.  BaseMethod calls model.gradient_checkpointing_enable()
     during build() if the surgery model exposes it.  Direct-wrap methods delegate
     to the HuggingFace model's own gradient checkpointing support.
"""

from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return-type contracts
# ---------------------------------------------------------------------------

@dataclass
class GeometricHealthMetrics:
    """
    Geometric health of a single weight matrix W, computed from its singular values.

    All metrics are scalars.  The per-head version is produced for every
    (layer, head) pair; the aggregated version averages across heads.

    Metrics defined precisely so subclasses and analysis scripts use identical math:

      sigma: singular values in descending order, shape [k], k = min(m, n)
      p_i = sigma_i / sum(sigma)

      stable_rank     = ||W||_F^2 / ||W||_2^2 = sum(sigma^2) / sigma_max^2
      sv_entropy      = -sum(p_i * log(p_i + eps))
      effective_rank  = exp(sv_entropy)              # Roy & Vetterli 2007
      condition_number = sigma_max / (sigma_min + eps)
      nuclear_norm    = sum(sigma)
      isotropy        = sigma_min / (sigma_max + eps)
    """
    stable_rank:      float = 0.0
    sv_entropy:       float = 0.0
    effective_rank:   float = 0.0
    condition_number: float = 0.0
    nuclear_norm:     float = 0.0
    isotropy:         float = 0.0

    @classmethod
    def from_singular_values(cls, sigma: torch.Tensor) -> "GeometricHealthMetrics":
        """
        Compute all six metrics from a 1-D tensor of singular values.
        sigma: descending order, all non-negative.
        """
        sigma = sigma.detach().float()   # fp32 for numerical stability
        sigma = sigma.clamp(min=0.0)     # guard tiny negatives from SVD
        eps = 1e-10

        s2   = sigma ** 2
        snuc = sigma.sum().item()
        smax = sigma[0].item() if sigma.numel() > 0 else eps
        smin = sigma[-1].item() if sigma.numel() > 0 else eps

        # stable rank
        sr = s2.sum().item() / (smax ** 2 + eps)

        # sv entropy & effective rank
        # clamp p *before* the log so log(0) is impossible even if a singular
        # value is exactly zero (collapsed weight matrix / dead head).
        # Using (p + eps).log() instead would underestimate entropy because the
        # eps shifts every probability up — clamping only the zeros is cleaner.
        p = sigma / (snuc + eps)
        p = p.clamp(min=eps)
        entropy = -(p * p.log()).sum().item()
        eff_rank = math.exp(entropy)

        # condition number & isotropy
        cond = smax / (smin + eps)
        iso  = smin / (smax + eps)

        return cls(
            stable_rank=sr,
            sv_entropy=entropy,
            effective_rank=eff_rank,
            condition_number=cond,
            nuclear_norm=snuc,
            isotropy=iso,
        )

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}stable_rank":      self.stable_rank,
            f"{prefix}sv_entropy":       self.sv_entropy,
            f"{prefix}effective_rank":   self.effective_rank,
            f"{prefix}condition_number": self.condition_number,
            f"{prefix}nuclear_norm":     self.nuclear_norm,
            f"{prefix}isotropy":         self.isotropy,
        }

    @classmethod
    def average(cls, metrics: List["GeometricHealthMetrics"]) -> "GeometricHealthMetrics":
        """Average a list of GeometricHealthMetrics (across heads or layers)."""
        if not metrics:
            return cls()
        fields = ["stable_rank", "sv_entropy", "effective_rank",
                  "condition_number", "nuclear_norm", "isotropy"]
        return cls(**{
            f: sum(getattr(m, f) for m in metrics) / len(metrics)
            for f in fields
        })


@dataclass
class PAFTSnapshot:
    """
    Tensors saved per training event for PAFT-specific post-training analysis.
    All tensors are CPU (moved before storing — no VRAM held by checkpoints).

    Shapes (GPT-2 small: 12 layers, 12 heads, d_head=64, n_embd=768):
        Q_V  : List[n_layers] of [n_heads, n_embd, d_head]   frozen semi-orthogonal
        Q_O  : List[n_layers] of [n_heads, d_head, n_embd]   frozen semi-orthogonal
        S_V  : List[n_layers] of [n_heads, d_head, d_head]   trainable scaling (hybrid)
        S_O  : List[n_layers] of [n_heads, d_head, d_head]   trainable scaling (hybrid)
        EV_V : List[n_layers] of [n_heads, d_head, d_head]   eigenvectors of S_V
        EV_O : List[n_layers] of [n_heads, d_head, d_head]   eigenvectors of S_O
        lam_V: List[n_layers] of [n_heads, d_head]            eigenvalues (pure only)
        lam_O: List[n_layers] of [n_heads, d_head]            eigenvalues (pure only)

    EV and lam are both populated for all PAFT variants because hybrid methods
    can also report them (via eigendecomposition of the current S at snapshot time).
    """
    Q_V:   List[torch.Tensor] = field(default_factory=list)
    Q_O:   List[torch.Tensor] = field(default_factory=list)
    S_V:   List[torch.Tensor] = field(default_factory=list)
    S_O:   List[torch.Tensor] = field(default_factory=list)
    EV_V:  List[torch.Tensor] = field(default_factory=list)
    EV_O:  List[torch.Tensor] = field(default_factory=list)
    lam_V: List[torch.Tensor] = field(default_factory=list)
    lam_O: List[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BaseMethod
# ---------------------------------------------------------------------------

class BaseMethod(ABC):
    """
    Abstract base class for all 11 fine-tuning methods.

    Subclass responsibilities:
        _build_model(hf_name) -> nn.Module   load model, apply weight surgery
        _configure_parameters()              set requires_grad correctly
        get_live_WV_WO() -> dict             live weight matrices for health metrics

    Optional overrides:
        paft_snapshot()       PAFT subclasses only — return PAFTSnapshot
        get_optimizer()       only if non-AdamW is genuinely needed
        pre_optimizer_step()  no method currently needs this
    """

    def __init__(self, method_name: str, cfg: dict):
        """
        Args:
            method_name: e.g. "pure_paft", "lora_r8" — used in logs and checkpoints
            cfg: the fully merged config dict for this experiment run
        """
        self.method_name = method_name
        self.cfg = cfg
        self.model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._built = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def build(self, hf_name: str, device: torch.device) -> None:
        """
        Load pretrained model, decompose, configure parameters.

        CPU offloading contract:
            _build_model() runs entirely on CPU.  Pretrained weights are naturally
            on CPU from HuggingFace.  Decomposition (SVD, polar) happens here,
            on CPU, so no VRAM is consumed during the O(n_layers * n_heads) pass.
            Only after decomposition does .to(device) move the needed components.

        Gradient checkpointing:
            Called before .to(device) so the checkpoint wrapper is in place before
            any CUDA allocations.  Trades activation recomputation for ~40% VRAM
            reduction on activations — essential for GPT-2 medium at 6 GB.

        Do NOT override build().  Override _build_model() and _configure_parameters().
        """
        self._device = device

        logger.info(f"[{self.method_name}] Building from {hf_name} ...")
        logger.info(f"[{self.method_name}] Decomposition pass on CPU ...")
        self.model = self._build_model(hf_name)

        # Enable gradient checkpointing before moving to device
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if self.cfg.get("training", {}).get("gradient_checkpointing", False):
                self.model.gradient_checkpointing_enable()
                logger.info(f"[{self.method_name}] Gradient checkpointing enabled.")
            else:
                logger.info(f"[{self.method_name}] Gradient checkpointing disabled -- using full activations.")

        logger.info(f"[{self.method_name}] Moving to {device} ...")
        self.model.to(device)

        # Freeze / unfreeze *after* move so requires_grad is set on device tensors
        self._configure_parameters()

        # _built must be True before calling any method that invokes _assert_built(),
        # including num_trainable_params() used in the log line below.
        self._built = True

        n_train = self.num_trainable_params()
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"[{self.method_name}] Ready — "
            f"{n_train:,} / {n_total:,} trainable "
            f"({100 * n_train / n_total:.3f}%)"
        )

    @abstractmethod
    def _build_model(self, hf_name: str) -> nn.Module:
        """
        Load pretrained weights and apply weight surgery if needed.

        Contract:
          - Run entirely on CPU (do not call .cuda() or .to(device)).
          - Return an nn.Module ready for .to(device) to be called on it.
          - Do NOT call _configure_parameters() here.
          - For direct-wrap methods (frozen, full_finetune, bitfit, lora):
              return GPT2LMHeadModel.from_pretrained(hf_name)
          - For surgery methods (PAFT, SVF):
              decompose pretrained weights on CPU, construct the custom model
              with frozen buffers and trainable parameters, return it.
        """

    @abstractmethod
    def _configure_parameters(self) -> None:
        """
        Set requires_grad correctly for this method.

        Called by build() after .to(device).  Must start by freezing everything,
        then selectively unfreeze.  Never start from an unfrozen state.

        Required pattern:
            freeze_all(self.model)
            for p in <method-specific trainable params>:
                p.requires_grad_(True)
        """

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single forward pass, returns scalar cross-entropy loss.

        Default delegates straight to self.model.  Surgery methods whose forward
        pass involves weight reconstruction (PAFT, SVF) implement this in their
        custom nn.Module, not here — so the default still applies.

        The trainer always calls method.forward(), never model() directly.
        """
        self._assert_built()
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def get_optimizer(self, lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        """
        Return AdamW over trainable parameters.

        All 11 methods use this default — including PoLAR.  PoLAR's landing
        field hooks modify .grad during backward(), so by the time AdamW reads
        the gradient it already has the Riemannian-corrected value.  AdamW then
        applies its moment estimates to the corrected gradient, which is the
        correct behaviour (Adam on the landing-field gradient).

        Override only if a genuinely different optimizer family is needed.
        """
        self._assert_built()
        return AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

    def pre_optimizer_step(self) -> None:
        """
        Hook called after loss.backward() and before optimizer.step().

        Default: no-op.  PoLAR's landing field fires during backward via
        register_hook — not here.  Kept as an explicit hook so the trainer
        loop can call it unconditionally without if-branches.
        """

    # ------------------------------------------------------------------
    # Parameter introspection
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """All parameters with requires_grad=True."""
        self._assert_built()
        return (p for p in self.model.parameters() if p.requires_grad)

    def num_trainable_params(self) -> int:
        """Total count of trainable scalars."""
        self._assert_built()
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def parameter_summary(self) -> Dict[str, dict]:
        """
        {name: {requires_grad, shape, numel}} for every named parameter.
        Used by tests/test_parameter_groups.py to verify freeze/unfreeze contracts.
        """
        self._assert_built()
        return {
            name: {
                "requires_grad": p.requires_grad,
                "shape":         list(p.shape),
                "numel":         p.numel(),
            }
            for name, p in self.model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Geometric health  (called every epoch for all 11 methods)
    # ------------------------------------------------------------------

    def geometric_health_snapshot(self) -> Dict[str, object]:
        """
        Compute the six geometric health metrics on live W_V and W_O for every
        (layer, head) pair.  Works identically for all methods because each
        implements get_live_WV_WO() which hides how the weight is stored.

        Computation is on CPU — tensors are moved off device before SVD to avoid
        allocating extra VRAM during an already memory-constrained training pass.

        Returns:
            {
                "per_head":  {layer: {head: {"W_V": GeometricHealthMetrics,
                                             "W_O": GeometricHealthMetrics}}},
                "per_layer": {layer: {"W_V": GeometricHealthMetrics (avg over heads),
                                      "W_O": GeometricHealthMetrics (avg over heads)}},
                "global":    {"W_V": GeometricHealthMetrics (avg over all layers/heads),
                              "W_O": GeometricHealthMetrics},
            }
        """
        self._assert_built()
        live     = self.get_live_WV_WO()
        n_layers = len(live["W_V"])
        n_heads  = live["W_V"][0].shape[0]

        per_head: Dict = {}
        all_V: List[GeometricHealthMetrics] = []
        all_O: List[GeometricHealthMetrics] = []

        with torch.no_grad():
            for l in range(n_layers):
                # Move layer tensors to CPU for SVD — frees VRAM immediately after
                W_V_l = live["W_V"][l].detach().cpu().float()  # [n_heads, n_embd, d_head]
                W_O_l = live["W_O"][l].detach().cpu().float()  # [n_heads, d_head, n_embd]

                layer_V: List[GeometricHealthMetrics] = []
                layer_O: List[GeometricHealthMetrics] = []
                per_head[l] = {}

                for h in range(n_heads):
                    sv_V = torch.linalg.svdvals(W_V_l[h])  # [d_head], descending
                    sv_O = torch.linalg.svdvals(W_O_l[h])  # [d_head], descending

                    m_V = GeometricHealthMetrics.from_singular_values(sv_V)
                    m_O = GeometricHealthMetrics.from_singular_values(sv_O)

                    per_head[l][h] = {"W_V": m_V, "W_O": m_O}
                    layer_V.append(m_V)
                    layer_O.append(m_O)

                all_V.extend(layer_V)
                all_O.extend(layer_O)

        per_layer = {
            l: {
                "W_V": GeometricHealthMetrics.average(
                    [per_head[l][h]["W_V"] for h in range(n_heads)]
                ),
                "W_O": GeometricHealthMetrics.average(
                    [per_head[l][h]["W_O"] for h in range(n_heads)]
                ),
            }
            for l in range(n_layers)
        }

        return {
            "per_head":  per_head,
            "per_layer": per_layer,
            "global": {
                "W_V": GeometricHealthMetrics.average(all_V),
                "W_O": GeometricHealthMetrics.average(all_O),
            },
        }

    @abstractmethod
    def get_live_WV_WO(self) -> Dict[str, List[torch.Tensor]]:
        """
        Return the actual live W_V and W_O weight matrices for every layer.

        Returns:
            {
                "W_V": List[n_layers] of Tensor[n_heads, n_embd, d_head],
                "W_O": List[n_layers] of Tensor[n_heads, d_head, n_embd],
            }

        Contract by method type:
            Additive (LoRA, PoLAR, BitFit):
                Reconstruct W_0 + ΔW.  For BitFit the weight matrices are
                unchanged — return W_0.  ΔW must be the current up-to-date ΔW,
                not a cached copy from a previous step.

            Surgery, non-additive (PAFT, SVF):
                Return the reconstructed weight from stored polar/SVD components.
                Must be mathematically identical to what the forward pass computes.
                PAFT: Q @ S (right polar) or S @ Q (left polar).
                SVF:  U @ diag(sigma) @ Vh.

            Frozen / full_finetune:
                Return the current model weight directly (same for both —
                full_finetune weights are simply moving during training).

        Tensors may be on any device.  geometric_health_snapshot() moves to CPU.
        Wrap body with torch.no_grad() — no gradients needed here.
        """

    # ------------------------------------------------------------------
    # PAFT-specific snapshot  (None for all baselines)
    # ------------------------------------------------------------------

    def paft_snapshot(self) -> Optional[PAFTSnapshot]:
        """
        Return a PAFTSnapshot for post-training geometric analysis.

        Default: None.  Only M8–M11 (PAFT variants) override this.

        The checkpointing schema calls this every epoch.  None is saved silently.
        All tensors in the returned snapshot must be on CPU.
        """
        return None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """
        Full state for checkpointing.  Sufficient to resume training or run analysis.
        Subclasses call super().state_dict() and add method-specific tensors.
        """
        self._assert_built()
        return {
            "method_name": self.method_name,
            "model":       self.model.state_dict(),
            "cfg":         self.cfg,
        }

    def load_state_dict(self, d: dict) -> None:
        """
        Restore from checkpoint.  build() must have been called first.
        """
        self._assert_built()
        if d.get("method_name") != self.method_name:
            raise ValueError(
                f"Checkpoint is for method '{d.get('method_name')}' but "
                f"current method is '{self.method_name}'"
            )
        self.model.load_state_dict(d["model"])

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "BaseMethod":
        """Move model to device and update internal reference."""
        self._assert_built()
        self._device = device
        self.model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        self._assert_built()
        return self._device

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Force-release VRAM and system RAM.  Call after every experiment run.

        Essential for the 8 GB VRAM constraint — without this, the next
        method's build() will OOM because the previous model is still resident.

        Sequence:
          1. Move model back to CPU before deletion so CUDA doesn't hold live
             references through the Python GC cycle.
          2. Delete the Python reference — triggers __del__ on the nn.Module
             and its Parameter tensors.
          3. gc.collect() to break any reference cycles CPython missed.
          4. torch.cuda.empty_cache() to return the now-freed CUDA blocks to
             the allocator pool (does NOT free memory the OS can reclaim, but
             makes it available for the next model).

        After cleanup(), _built is False.  build() must be called again before
        the method can be used.
        """
        if self.model is not None:
            self.model.to("cpu")   # cut CUDA references before del
            del self.model
            self.model = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._built = False
        logger.info(f"[{self.method_name}] Cleanup complete. VRAM released.")

    def _assert_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                f"{self.__class__.__name__} must be built before use. "
                f"Call {self.__class__.__name__}.build(hf_name, device) first."
            )

    def __repr__(self) -> str:
        if self._built:
            return (
                f"{self.__class__.__name__}("
                f"{self.method_name!r}, "
                f"trainable={self.num_trainable_params():,})"
            )
        return f"{self.__class__.__name__}({self.method_name!r}, not built)"


# ---------------------------------------------------------------------------
# Module-level utilities used by every subclass
# ---------------------------------------------------------------------------

def freeze_all(model: nn.Module) -> None:
    """Freeze every parameter.  Always the first call in _configure_parameters()."""
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze(params) -> None:
    """Unfreeze an iterable of nn.Parameter."""
    for p in params:
        p.requires_grad_(True)


def svdvals_of(W: torch.Tensor) -> torch.Tensor:
    """
    Descending singular values of a 2-D weight matrix.
    Thin wrapper around torch.linalg.svdvals — centralised so analysis scripts
    and geometric_health_snapshot() use the exact same computation.
    """
    return torch.linalg.svdvals(W)   # torch guarantees descending order
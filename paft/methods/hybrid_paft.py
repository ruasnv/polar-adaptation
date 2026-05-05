"""
hybrid_paft: train full S_V and S_O matrices (unconstrained d_head x d_head).

Parameterization:
  W_V_h = Q_V_h @ S_V_h   (Q frozen, S directly trainable — no symmetry constraint)
  W_O_h = S_O_h @ Q_O_h

S is initialized from the polar decomposition (symmetric PSD) but is free
to evolve during training. The rotation constraint is preserved (Q frozen),
but S can break symmetry as adaptation proceeds.

Parameter count:
  GPT-2 small:  12 * 12 * 2 * 64 * 64 = 1,179,648
  GPT-2 medium: 24 * 16 * 2 * 64 * 64 = 3,145,728
"""

from typing import Any, Dict
from paft.methods.base import BaseMethod
from paft.model.paft_model import PAFTModel
from paft.model.parameter_groups import configure_hybrid_paft
from paft.decomposition.geometry import effective_rank


class HybridPAFT(BaseMethod):

    def _build_model(self, base_model) -> PAFTModel:
        return PAFTModel(base_model)

    def _configure_parameters(self) -> None:
        configure_hybrid_paft(self.model)

    def compute_adaptation_metrics(self) -> Dict[str, Any]:
        """
        Tracks S matrix statistics per layer.
        Rotation drift is 0 (Q is frozen); S can change in any direction.
        """
        metrics: Dict[str, Any] = {}
        for l, layer in enumerate(self.model.paft_layers()):
            S_V = layer.S_V.detach()   # [n_heads, d_head, d_head]
            S_O = layer.S_O.detach()

            # Mean effective rank across heads
            rank_V = sum(effective_rank(S_V[h]) for h in range(layer.n_heads)) / layer.n_heads
            rank_O = sum(effective_rank(S_O[h]) for h in range(layer.n_heads)) / layer.n_heads

            metrics[f"layer_{l}"] = {
                "S_V_frobenius_mean": S_V.norm(p='fro', dim=(-2, -1)).mean().item(),
                "S_O_frobenius_mean": S_O.norm(p='fro', dim=(-2, -1)).mean().item(),
                "effective_rank_V_mean": rank_V,
                "effective_rank_O_mean": rank_O,
                # Q is frozen — rotation drift is identically zero
                "rotation_drift_V":  0.0,
                "rotation_drift_O":  0.0,
            }
        return metrics
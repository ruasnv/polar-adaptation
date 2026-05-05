from __future__ import annotations
from typing import Dict, Type, Any

from paft.methods.base import BaseMethod

# Import all method classes
from paft.methods.baselines.frozen import Frozen
from paft.methods.baselines.full_finetune import FullFinetune
from paft.methods.baselines.bitfit import BitFit
from paft.methods.baselines.lora import LoRABaseline
from paft.methods.baselines.svf import SVFBaseline
from paft.methods.baselines.polar import PolarBaseline
from paft.methods.pure_paft import PurePAFT
from paft.methods.hybrid_paft import HybridPAFT
from paft.methods.safe_pure_paft import SafePurePAFT
from paft.methods.safe_hybrid_paft import SafeHybridPAFT

# Map the YAML 'name' to the actual Python Class
METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {
    "frozen": Frozen,
    "full_finetune": FullFinetune,
    "bitfit": BitFit,
    "lora": LoRABaseline,
    "svf": SVFBaseline,
    "polar": PolarBaseline,
    "pure_paft": PurePAFT,
    "hybrid_paft": HybridPAFT,
    "safe_pure_paft": SafePurePAFT,
    "safe_hybrid_paft": SafeHybridPAFT,
}


def get_method_instance(method_name: str, cfg: Dict[str, Any]) -> BaseMethod:
    """
    Factory method to instantiate a tuning strategy.

    Args:
        method_name: The string from your YAML (e.g., 'safe_hybrid_paft')
        cfg: The full merged config dict.
    """
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Method '{method_name}' not found in registry. "
                         f"Available: {list(METHOD_REGISTRY.keys())}")

    method_class = METHOD_REGISTRY[method_name]
    return method_class(method_name=method_name, cfg=cfg)
"""
Method registry — single source of truth for all 11 fine-tuning methods.

Keys match the YAML method name field exactly:
    cfg = get_config("gpt2_small", "news", "lora_r8")
    method = get_method("lora_r8", cfg)

Both lora_r8 and lora_r64 map to LoRABaseline — the class reads
rank/alpha from cfg['method']['lora'] set by the respective YAML file.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from paft.methods.base import BaseMethod
from paft.methods.pure_paft        import PurePAFT
from paft.methods.hybrid_paft      import HybridPAFT
from paft.methods.safe_pure_paft   import SafePurePAFT
from paft.methods.safe_hybrid_paft import SafeHybridPAFT
from paft.methods.baselines.frozen        import Frozen
from paft.methods.baselines.full_finetune import FullFinetune
from paft.methods.baselines.bitfit        import BitFit
from paft.methods.baselines.lora          import LoRABaseline
from paft.methods.baselines.svf           import SVFBaseline
from paft.methods.baselines.polar         import PolarBaseline


METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {
    # Bounds
    "frozen":           Frozen,
    "full_finetune":    FullFinetune,
    # Additive baselines — both map to LoRABaseline; rank/alpha come from YAML
    "lora_r8":          LoRABaseline,
    "lora_r64":         LoRABaseline,
    # Geometric baselines
    "bitfit":           BitFit,
    "svf":              SVFBaseline,
    "polar":            PolarBaseline,
    # PAFT variants
    "pure_paft":        PurePAFT,
    "hybrid_paft":      HybridPAFT,
    "safe_pure_paft":   SafePurePAFT,
    "safe_hybrid_paft": SafeHybridPAFT,
}


def get_method(method_name: str, cfg: Dict[str, Any]) -> BaseMethod:
    """
    Instantiate a method by name.

    Does NOT call build() — the caller is responsible for calling
    method.build(hf_name, device) after instantiation so that
    the CPU-offloaded decomposition and device placement are
    controlled by the trainer, not hidden inside the factory.

    Args:
        method_name: Key from METHOD_REGISTRY (matches YAML name field).
        cfg:         Fully merged config dict from get_config().

    Returns:
        Uninitialised BaseMethod instance (build() not yet called).

    Raises:
        ValueError if method_name is not registered.
    """
    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Registered: {sorted(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[method_name](method_name=method_name, cfg=cfg)


def list_methods() -> list[str]:
    """Return sorted list of all registered method names."""
    return sorted(METHOD_REGISTRY.keys())
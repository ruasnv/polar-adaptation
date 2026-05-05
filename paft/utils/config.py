"""
Config loading with YAML inheritance.

Merge order (later overrides earlier):
    base.yaml  ←  parent method yaml  ←  child method yaml  ←  model yaml

The 'extends' key in a method YAML triggers recursive parent resolution before
the final merge, so safe_hybrid_paft correctly inherits hybrid_paft's structure
without duplicating every freeze/tune entry.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a NEW dict that is base deep-merged with override.

    Neither input dict is mutated.  Nested dicts are merged recursively;
    all other value types are replaced by the override value.

    This is a pure function — safe to call multiple times with the same base
    across a sweep without accumulating state between experiments.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ConfigLoader:
    """
    Loads and merges experiment configs from the configs/ directory.

    Resolves 'extends' inheritance recursively for method configs.
    Merges: base.yaml + method chain + model yaml (in that order).

    Usage:
        loader = ConfigLoader()
        cfg = loader.load(model="gpt2_small", domain="news", method="safe_hybrid_paft")
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        if project_root is None:
            # paft/utils/config.py → paft/utils → paft → project root
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root)

        self.config_dir = self.project_root / "configs"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        model:  str,
        domain: str,
        method: str,
    ) -> Dict[str, Any]:
        """
        Build the final merged config for one experiment.

        Returns a single flat-ish dict that the trainer and method instances
        consume.  Keys: 'training', 'logging', 'geometric_health', 'model',
        'domain', 'method'.
        """
        base_cfg   = self._load_yaml("base.yaml")
        model_cfg  = self._load_yaml(f"models/{model}.yaml")
        domain_cfg = self._load_yaml(f"domains/{domain}.yaml")
        method_cfg = self._resolve_method_inheritance(method)

        # Merge order: base ← model ← domain ← method
        cfg = deep_update(base_cfg,   model_cfg)
        cfg = deep_update(cfg,        domain_cfg)
        cfg = deep_update(cfg,        method_cfg)

        # Stamp the experiment identity into the config
        cfg["experiment"] = {
            "model":  model,
            "domain": domain,
            "method": method,
        }

        logger.debug(
            f"Config loaded: model={model} domain={domain} method={method}"
        )
        return cfg

    def load_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        Load only the method chain merged on top of base.yaml.
        Used by analysis scripts that don't need model/domain context.
        """
        base_cfg   = self._load_yaml("base.yaml")
        method_cfg = self._resolve_method_inheritance(method_name)
        return deep_update(base_cfg, method_cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, relative_path: str) -> Dict[str, Any]:
        path = self.config_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"(config_dir={self.config_dir})"
            )
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def _resolve_method_inheritance(self, method_name: str) -> Dict[str, Any]:
        """
        Recursively resolve 'extends' chains.

        Example: safe_hybrid_paft extends hybrid_paft
            → load hybrid_paft.yaml  (no extends)
            → load safe_hybrid_paft.yaml
            → merge: hybrid_paft ← safe_hybrid_paft overrides
        """
        current = self._load_yaml(f"methods/{method_name}.yaml")

        if "extends" in current:
            parent_name = current.pop("extends")
            logger.info(f"Config inheritance: {method_name} extends {parent_name}")
            parent = self._resolve_method_inheritance(parent_name)
            return deep_update(parent, current)

        return current


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_config(
    model:  str,
    domain: str,
    method: str,
    project_root: str | Path | None = None,
) -> Dict[str, Any]:
    """
    One-liner for scripts.

    Example:
        cfg = get_config("gpt2_small", "news", "hybrid_paft")
    """
    return ConfigLoader(project_root).load(model, domain, method)
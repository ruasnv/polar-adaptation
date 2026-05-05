from __future__ import annotations

import logging
import yaml
import log_utils
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively updates a dictionary.
    Ensures nested blocks like 'method.freeze' are merged, not overwritten.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


class ConfigLoader:
    """
    Handles YAML loading with support for hierarchical inheritance.

    Structure:
    1. Loads configs/base.yaml (Global Defaults)
    2. Loads configs/methods/{method}.yaml (The Leaf)
    3. If leaf has 'extends', recursively loads and merges parents.
    """

    def __init__(self, project_root: str | Path | None = None):
        if project_root is None:
            # Assume we are in paft/utils/, so project root is ../../
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root)

        self.config_dir = self.project_root / "configs"
        self.methods_dir = self.config_dir / "methods"

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def load_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        The main entry point for the trainer.
        Example: loader.load_method_config("safe_hybrid_paft")
        """
        # 1. Load the core defaults
        base_cfg = self._load_yaml(self.config_dir / "base.yaml")

        # 2. Load the method-specific chain
        method_cfg = self._resolve_inheritance(method_name)

        # 3. Final Merge: Method overrides Base Defaults
        final_cfg = deep_update(base_cfg, method_cfg)

        return final_cfg

    def _resolve_inheritance(self, method_name: str) -> Dict[str, Any]:
        """
        Recursively resolves the 'extends' keyword.
        """
        path = self.methods_dir / f"{method_name}.yaml"
        current_cfg = self._load_yaml(path)

        if "extends" in current_cfg:
            parent_name = current_cfg.pop("extends")
            logger.info(f"Config: {method_name} extends {parent_name}")

            # Recursive call to get parent (and its parents)
            parent_cfg = self._resolve_inheritance(parent_name)

            # Merge current (leaf) into parent
            return deep_update(parent_cfg, current_cfg)

        return current_cfg


def get_config(method_name: str) -> Dict[str, Any]:
    """Utility helper for scripts."""
    loader = ConfigLoader()
    return loader.load_method_config(method_name)
"""
Inference configuration loader.

Loads and validates inference parameters from YAML config file.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "inference.yaml"


class InferenceConfig:
    """Loads and manages inference configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Load inference config from YAML file.

        Args:
            config_path: Path to config YAML. If None, uses default configs/inference.yaml
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    @property
    def checkpoint_path(self) -> str:
        """Path to model checkpoint."""
        return self._config.get("checkpoint", {}).get("path", "output/models/best.pt")

    @property
    def device(self) -> str:
        """Inference device (cpu, cuda, mps)."""
        return self._config.get("inference", {}).get("device", "cpu")

    @property
    def target_return(self) -> float:
        """Target return-to-go for conditioning."""
        return float(self._config.get("inference", {}).get("target_return", 3.0))

    @property
    def temperature(self) -> float:
        """Sampling temperature."""
        return float(self._config.get("inference", {}).get("temperature", 1.5))

    @property
    def fallback_tile_x(self) -> int:
        """Default tile x-coordinate for fallback."""
        return int(self._config.get("strategy", {}).get("fallback_tile_x", 9))

    @property
    def fallback_tile_y(self) -> int:
        """Default tile y-coordinate for fallback."""
        return int(self._config.get("strategy", {}).get("fallback_tile_y", 24))

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return self._config.copy()

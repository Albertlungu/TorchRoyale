"""
Utility functions for TorchRoyale GUI.
"""

import os

import yaml

from src.constants import SRC_DIR


def load_config() -> dict:
    """Load configuration from the config.yaml file."""
    try:
        config_path = os.path.join(SRC_DIR, "config.yaml")
        with open(config_path, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception("Can't parse config.") from e


def save_config(config: dict) -> None:
    """Save configuration to the config.yaml file."""
    try:
        config_path = os.path.join(SRC_DIR, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(config, file)
    except Exception as e:
        raise Exception("Can't save config.") from e

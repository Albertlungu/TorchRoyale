"""Utility functions for the TorchRoyale desktop UI."""

from pathlib import Path
from typing import Any
from typing import Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "app_config.yaml"
DEFAULT_CONFIG: dict[str, Any] = {
    "adb": {"ip": "127.0.0.1", "device_serial": ""},
    "bot": {
        "auto_start_game": False,
        "load_deck": False,
        "log_level": "INFO",
    },
    "ingame": {"play_action": 1.0},
    "visuals": {
        "save_images": False,
        "save_labels": False,
        "show_images": False,
    },
}


def _merge_defaults(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge user config with default config values.

    Args:
        config (Optional[dict[str, Any]]): User-provided config to merge (may be None).

    Returns:
        dict[str, Any]: Merged config with defaults filled in where missing.
    """
    merged: dict[str, Any] = {
        section: values.copy() if isinstance(values, dict) else values
        for section, values in DEFAULT_CONFIG.items()
    }
    if not config:
        return merged
    for section, values in config.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def load_config() -> dict[str, Any]:
    """
    Load UI config from `configs/app_config.yaml`.

    Args:
        None

    Returns:
        dict[str, Any]: Configuration dictionary with defaults merged in.
    """
    if not CONFIG_PATH.exists():
        return _merge_defaults(None)
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return _merge_defaults(yaml.safe_load(file))


def save_config(config: dict[str, Any]) -> None:
    """
    Persist UI config to `configs/app_config.yaml`.

    Args:
        config (dict[str, Any]): Configuration dictionary to save.
    Returns:
        None
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(_merge_defaults(config), file, sort_keys=True)

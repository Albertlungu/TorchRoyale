"""Utility functions for the TorchRoyale desktop UI."""

from pathlib import Path
from typing import Optional
from typing import TypedDict

import yaml


class AdbConfig(TypedDict):
    ip: str
    device_serial: str


class BotConfig(TypedDict):
    auto_start_game: bool
    load_deck: bool
    log_level: str


class IngameConfig(TypedDict):
    play_action: float


class VisualsConfig(TypedDict):
    save_images: bool
    save_labels: bool
    show_images: bool


class AppConfig(TypedDict):
    adb: AdbConfig
    bot: BotConfig
    ingame: IngameConfig
    visuals: VisualsConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "app_config.yaml"
DEFAULT_CONFIG: AppConfig = {
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


def _merge_defaults(config: Optional[AppConfig]) -> AppConfig:
    """
    Merge user config with default config values.

    Args:
        config (Optional[AppConfig]): User-provided config to merge (may be None).

    Returns:
        AppConfig: Merged config with defaults filled in where missing.
    """
    merged: AppConfig = {
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


def load_config() -> AppConfig:
    """
    Load UI config from `configs/app_config.yaml`.

    Args:
        None

    Returns:
        AppConfig: Configuration dictionary with defaults merged in.
    """
    if not CONFIG_PATH.exists():
        return _merge_defaults(None)
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return _merge_defaults(yaml.safe_load(file))


def save_config(config: AppConfig) -> None:
    """
    Persist UI config to `configs/app_config.yaml`.

    Args:
        config (AppConfig): Configuration dictionary to save.
    Returns:
        None
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(_merge_defaults(config), file, sort_keys=True)

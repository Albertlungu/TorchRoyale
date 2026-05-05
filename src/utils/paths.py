"""Helpers for locating bundled resources and writable app data."""

from __future__ import annotations

import sys
from pathlib import Path


APP_NAME = "TorchRoyale"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def is_frozen() -> bool:
    """
    Return whether the app is running from a frozen PyInstaller bundle.

    Returns:
        bool: ``True`` when running from a frozen bundle, ``False`` otherwise.
    """
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """
    Return the root directory that contains bundled read-only assets.

    Returns:
        Path: PyInstaller ``_MEIPASS`` directory when frozen, otherwise the
            repository root.
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return _REPO_ROOT


def writable_root() -> Path:
    """
    Return the root directory for writable user data.

    Returns:
        Path: Platform-specific app-data directory when frozen (created if
            absent), otherwise the repository root.
    """
    if not is_frozen():
        return _REPO_ROOT

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"

    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource_path(*parts: str) -> Path:
    """
    Build a path inside the resource bundle or repo checkout.

    Args:
        *parts (str): Path components to join under the resource root.

    Returns:
        Path: Absolute path to the requested resource.
    """
    return resource_root().joinpath(*parts)


def writable_path(*parts: str) -> Path:
    """
    Build a path inside the writable app data directory.

    Args:
        *parts (str): Path components to join under the writable root.

    Returns:
        Path: Absolute path to the requested writable location.
    """
    return writable_root().joinpath(*parts)

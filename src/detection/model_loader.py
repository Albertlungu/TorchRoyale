"""Helpers for resolving detector model paths."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "data" / "models"


def resolve_model_path(model_name: str) -> Path:
    """Return the absolute path for a model and validate that it exists."""
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return model_path


def preferred_unit_model_path() -> Path:
    """Return the preferred unit detector model path.

    `best.mlpackage` is preferred when present, but the current migrated
    detector implementation uses the shipped ONNX model as the executable
    fallback until a Core ML runtime adapter is added.
    """
    coreml_path = MODELS_DIR / "best.mlpackage"
    if coreml_path.exists():
        return coreml_path
    return resolve_model_path("units_M_480x352.onnx")

"""
Train YOLOv8s on the TorchRoyale enemies-only dataset for opponent cards detection.

Trains a YOLOv8s model on data/datasets/torchroyale-enemies and saves the best
weights to data/models/onfield/enemies_best.pt.

Usage:
  python training/train_enemies.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_ENEMIES_DATASET = _PROJECT_ROOT / "data" / "datasets" / "torchroyale-enemies"
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "models" / "onfield"
_BEST_WEIGHTS_NAME = "torchroyale-enemies-best.pt"


def _prepare_data_yaml(data_yaml: Path) -> Path:
    """
    Create a resolved dataset YAML with valid split paths.

    Roboflow exports may reference ../valid and ../test while only train/ exists.
    In that case, val/test are pointed to train/images so Ultralytics can start.
    """
    text = data_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()

    train_images = _ENEMIES_DATASET / "train" / "images"
    valid_images = _ENEMIES_DATASET / "valid" / "images"
    val_images = _ENEMIES_DATASET / "val" / "images"
    test_images = _ENEMIES_DATASET / "test" / "images"

    if not train_images.exists():
        print(f"Error: missing train images directory: {train_images}")
        sys.exit(1)

    resolved_train = train_images
    resolved_val = (
        val_images
        if val_images.exists()
        else (valid_images if valid_images.exists() else train_images)
    )
    resolved_test = test_images if test_images.exists() else resolved_val

    updated = []
    seen_train = False
    seen_val = False
    seen_test = False
    for ln in lines:
        s = ln.lstrip()
        if s.startswith("train:"):
            updated.append(f"train: {resolved_train}")
            seen_train = True
        elif s.startswith("val:"):
            updated.append(f"val: {resolved_val}")
            seen_val = True
        elif s.startswith("test:"):
            updated.append(f"test: {resolved_test}")
            seen_test = True
        else:
            updated.append(ln)

    if not seen_train:
        updated.insert(0, f"train: {resolved_train}")
    if not seen_val:
        updated.insert(1, f"val: {resolved_val}")
    if not seen_test:
        updated.insert(2, f"test: {resolved_test}")

    prepared_yaml = _ENEMIES_DATASET / "data.resolved.yaml"
    prepared_yaml.write_text("\n".join(updated) + "\n", encoding="utf-8")

    print(f"Dataset YAML: {prepared_yaml}")
    print(f"  train -> {resolved_train}")
    print(f"  val   -> {resolved_val}")
    print(f"  test  -> {resolved_test}")
    return prepared_yaml


def _detect_device() -> str:
    """
    Auto-detect the best available PyTorch device.

    Returns:
        Device string: "mps", "cuda", or "cpu".
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    """Train YOLOv8s on the TorchRoyale enemies-only dataset."""
    print("=" * 60)
    print("TorchRoyale Enemies YOLOv8s Training")
    print("=" * 60)

    # Verify dataset exists
    data_yaml = _ENEMIES_DATASET / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: Dataset not found at {_ENEMIES_DATASET}")
        sys.exit(1)
    resolved_data_yaml = _prepare_data_yaml(data_yaml)

    # Verify and print class information
    print("\nDataset Classes:")
    import yaml
    with open(resolved_data_yaml) as f:
        data_config = yaml.safe_load(f)
    classes = data_config.get("names", [])
    print(f"  Total classes: {len(classes)}")
    print(f"  Sample classes: {classes[:5]} ... {classes[-5:]}")
    if "hero-knight" in classes:
        idx = classes.index("hero-knight")
        print(f"  ✓ 'hero-knight' found at index {idx}")
    else:
        print(f"  ✗ WARNING: 'hero-knight' NOT found in classes!")
        print(f"  Knights in dataset: {[c for c in classes if 'knight' in c.lower()]}")

    # Detect device
    device = _detect_device()
    print(f"\nUsing device: {device}")
    batch = 32 if device == "cuda" else 16

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)

    # Create output directory
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and train
    print("Loading YOLOv8s pre-trained weights...")
    model = YOLO("yolov8s.pt")

    print(f"Starting training on {resolved_data_yaml}...")
    results = model.train(
        data=str(resolved_data_yaml),
        epochs=150,
        imgsz=416,
        batch=batch,
        device=device,
        project=str(_OUTPUT_DIR),
        name="enemies-detector",
        save=True,
        save_period=2,
        verbose=True,
        # Heavy color augmentation to close the domain gap between
        # Roboflow exports and replay captures.
        hsv_h=0.05,
        hsv_s=0.9,
        hsv_v=0.6,
    )

    # Print metrics
    print("\n" + "=" * 60)
    print("Training Complete - Metrics:")
    print("=" * 60)

    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        print(f"mAP@50: {map50:.4f}")

    # Try to get per-class AP from validation results
    try:
        val_results = model.val(data=str(resolved_data_yaml), device=device)
        if hasattr(val_results, "box"):
            print("\nPer-class AP@50:")
            names = model.names if hasattr(model, "names") else {}
            for cls_idx, ap in enumerate(getattr(val_results.box, "ap50", [])):
                cls_name = names.get(cls_idx, f"class_{cls_idx}")
                print(f"  {cls_name}: {ap:.4f}")
    except Exception as e:
        print(f"Could not compute per-class AP: {e}")

    # Copy best.pt to canonical path
    best_source = _OUTPUT_DIR / "torchroyale-enemies" / "weights" / "best.pt"
    best_dest = _OUTPUT_DIR / _BEST_WEIGHTS_NAME

    if best_source.exists():
        shutil.copy2(best_source, best_dest)
        print(f"\nBest weights copied to: {best_dest}")
    else:
        print(f"Warning: best.pt not found at {best_source}")
        print("Check training output directory.")

    print("\nDone.")


if __name__ == "__main__":
    main()

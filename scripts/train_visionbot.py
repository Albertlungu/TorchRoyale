"""
Train YOLOv8s on the Vision Bot all-enemy-cards dataset for opponent cards detection.

Trains a YOLOv8s model on the Vision Bot dataset (opponent's cards on the field)
and saves the best weights to data/models/onfield/visionbot_best.pt.

Usage:
  python scripts/train_visionbot.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_VISIONBOT_DATASET = _PROJECT_ROOT / "data" / "datasets" / "visionbot_enemy"
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "models" / "onfield"
_BEST_WEIGHTS_NAME = "visionbot_best.pt"


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
    """Train YOLOv8n on the Vision Bot all-enemy-cards dataset."""
    print("=" * 60)
    print("Vision Bot YOLOv8s Training (All Enemy Cards)")
    print("=" * 60)

    # Verify dataset exists
    data_yaml = _VISIONBOT_DATASET / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: Dataset not found at {_VISIONBOT_DATASET}")
        print("Please run: python scripts/download_datasets.py")
        sys.exit(1)

    # Detect device
    device = _detect_device()
    print(f"Using device: {device}")
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

    print(f"Starting training on {data_yaml}...")
    results = model.train(
        data=str(data_yaml),
        epochs=200,
        imgsz=640,
        batch=batch,
        device=device,
        project=str(_OUTPUT_DIR),
        name="visionbot",
        save=True,
        save_period=2,
        verbose=True,
        # Heavy color augmentation to close the domain gap between
        # Roboflow (blue arena) and our videos (varied arena colors).
        hsv_h=0.05,
        hsv_s=0.9,
        hsv_v=0.6,
    )

    # Print metrics
    print("\n" + "=" * 60)
    print("Training Complete - Metrics:")
    print("=" * 60)

    # Access the results properly
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        print(f"mAP@50: {map50:.4f}")

    # Try to get per-class AP from validation results
    try:
        val_results = model.val(data=str(data_yaml), device=device)
        if hasattr(val_results, "box"):
            print("\nPer-class AP@50:")
            names = model.names if hasattr(model, "names") else {}
            for cls_idx, ap in enumerate(getattr(val_results.box, "ap50", [])):
                cls_name = names.get(cls_idx, f"class_{cls_idx}")
                print(f"  {cls_name}: {ap:.4f}")
    except Exception as e:
        print(f"Could not compute per-class AP: {e}")

    # Copy best.pt to canonical path
    best_source = _OUTPUT_DIR / "visionbot" / "weights" / "best.pt"
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

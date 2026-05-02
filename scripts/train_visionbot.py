"""
Train an on-field detector on the Vision Bot enemy-card dataset.

Defaults are tuned for Clash Royale's small, cluttered arena objects:
- YOLO11m baseline instead of YOLOv8n
- higher input resolution for better small-object recall
- optional YOLO11l fallback when more accuracy is needed

Usage:
  python scripts/train_visionbot.py
  python scripts/train_visionbot.py --model yolo11l.pt --imgsz 1280
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_VISIONBOT_DATASET = _PROJECT_ROOT / "data" / "datasets" / "visionbot_enemy"
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "models" / "onfield"
_BEST_WEIGHTS_NAME = "visionbot_best.pt"
_DEFAULT_MODEL = "yolo11m.pt"
_DEFAULT_IMGSZ = 960
_DEFAULT_EPOCHS = 100
_DEFAULT_BATCH = 16


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train the Vision Bot on-field detector.",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="Ultralytics checkpoint to fine-tune, e.g. yolo11m.pt or yolo11l.pt.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=_DEFAULT_IMGSZ,
        help="Training image size. Use 960 by default; try 1280 for harder small-object cases.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=_DEFAULT_BATCH,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional Ultralytics run name. Defaults to a model/imgsz-based name.",
    )
    return parser.parse_args()


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


def _default_run_name(model_name: str, imgsz: int) -> str:
    """Build a stable run name from the chosen model and image size."""
    stem = Path(model_name).stem.replace(".", "_")
    return f"visionbot-{stem}-{imgsz}"


def main() -> None:
    """Train the Vision Bot all-enemy-cards detector."""
    args = _parse_args()
    run_name = args.run_name or _default_run_name(args.model, args.imgsz)

    print("=" * 60)
    print("Vision Bot On-Field Detector Training (All Enemy Cards)")
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

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)

    # Create output directory
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and train
    print(f"Loading pre-trained weights: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on {data_yaml}...")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(_OUTPUT_DIR),
        name=run_name,
        save=True,
        save_period=1,
        verbose=True,
    )

    # Print metrics
    print("\n" + "=" * 60)
    print("Training Complete - Metrics:")
    print("=" * 60)

    # Access the results properly
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        recall = metrics.get("metrics/recall(B)", 0.0)
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        print(f"Recall: {recall:.4f}")
        print(f"mAP@50: {map50:.4f}")
        print(f"mAP@50-95: {map50_95:.4f}")

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
    best_source = _OUTPUT_DIR / run_name / "weights" / "best.pt"
    best_dest = _OUTPUT_DIR / _BEST_WEIGHTS_NAME

    if best_source.exists():
        shutil.copy2(best_source, best_dest)
        print(f"\nBest weights copied to: {best_dest}")
        model_specific_dest = _OUTPUT_DIR / f"{Path(args.model).stem}_{args.imgsz}_best.pt"
        shutil.copy2(best_source, model_specific_dest)
        print(f"Model-specific best weights copied to: {model_specific_dest}")
    else:
        print(f"Warning: best.pt not found at {best_source}")
        print("Check training output directory.")

    print("\nDone.")


if __name__ == "__main__":
    main()

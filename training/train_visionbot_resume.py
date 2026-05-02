"""
Resume Vision Bot YOLOv8 training from the last visionbot-2 checkpoint.

Loads data/models/onfield/visionbot-2/weights/last.pt, resumes training with the
same Vision Bot dataset, and copies the resulting best weights to the canonical
data/models/onfield/visionbot_best.pt path.

Usage:
  python scripts/train_visionbot_resume.py
  python scripts/train_visionbot_resume.py --epochs 150 --name visionbot-3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_VISIONBOT_DATASET = _PROJECT_ROOT / "data" / "datasets" / "visionbot_enemy"
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "models" / "onfield"
_LAST_CHECKPOINT = _OUTPUT_DIR / "visionbot-2" / "weights" / "last.pt"
_BEST_WEIGHTS_NAME = "visionbot_best.pt"


def _detect_device() -> str:
    """Auto-detect the best available PyTorch device."""
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume Vision Bot YOLOv8 training from the visionbot-2 last checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(_LAST_CHECKPOINT),
        help="Checkpoint to resume from.",
    )
    parser.add_argument(
        "--data",
        default=str(_VISIONBOT_DATASET / "data.yaml"),
        help="Dataset YAML path.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Target epoch count.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device. Defaults to auto-detected MPS/CUDA/CPU.",
    )
    parser.add_argument(
        "--project",
        default=str(_OUTPUT_DIR),
        help="Ultralytics project directory.",
    )
    parser.add_argument(
        "--name",
        default="visionbot-2",
        help="Ultralytics run name. Use a new name if you want a separate run folder.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing run directory when not resuming in-place.",
    )
    return parser.parse_args()


def main() -> None:
    """Resume YOLOv8 training on the Vision Bot dataset."""
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    data_yaml = Path(args.data)
    device = args.device or _detect_device()

    print("=" * 60)
    print("Vision Bot YOLOv8 Resume Training")
    print("=" * 60)

    if not checkpoint.exists():
        print(f"Error: Checkpoint not found at {checkpoint}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Resuming from: {checkpoint}")
    print(f"Dataset: {data_yaml}")

    model = YOLO(str(checkpoint))
    # resume=True tells Ultralytics to read all training args from the
    # checkpoint's saved args.yaml. Passing data/epochs/name alongside
    # resume=True causes a class-count mismatch in the TAL assigner because
    # the model head stays frozen to the checkpoint's nc while the data
    # loader uses the new dataset's nc.
    results = model.train(
        resume=True,
        device=device,
        batch=args.batch,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training Complete - Metrics:")
    print("=" * 60)

    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        print(f"mAP@50: {map50:.4f}")

    run_dir = project_dir / args.name
    best_source = run_dir / "weights" / "best.pt"
    best_dest = project_dir / _BEST_WEIGHTS_NAME

    if best_source.exists():
        shutil.copy2(best_source, best_dest)
        print(f"\nBest weights copied to: {best_dest}")
    else:
        print(f"Warning: best.pt not found at {best_source}")
        print("Check training output directory.")

    print("\nDone.")


if __name__ == "__main__":
    main()

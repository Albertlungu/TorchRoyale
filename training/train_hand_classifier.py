#!/usr/bin/env python3
"""
Train the YOLOv8 hand-card classifier on the downloaded Roboflow dataset.

Writes the best weights to data/models/hand_classifier/weights/best.pt.

Usage:
  python scripts/train_hand_classifier.py
  python scripts/train_hand_classifier.py --epochs 30 --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 hand card classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--data",
        default="data/datasets/hand_classifier_dataset",
        help="Path to the downloaded Roboflow dataset folder.",
    )
    parser.add_argument(
        "--output",
        default="data/models/hand_classifier",
        help="Directory where weights will be saved.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset not found at {data_path}")
        print("  Run the Roboflow download first:")
        print('  python -c "from roboflow import Roboflow; ..."')
        sys.exit(1)

    train_dir = data_path / "train"
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    total_images = sum(len(list((train_dir / cls).iterdir())) for cls in classes)
    print(f"Dataset:   {data_path}")
    print(f"Classes:   {len(classes)}")
    print(f"Train images: {total_images}")
    print(
        f"Epochs:    {args.epochs}  |  imgsz: {args.imgsz}  |  batch: {args.batch}  |  device: {args.device}"
    )
    print("-" * 60)

    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    model = YOLO("yolov8n-cls.pt")
    print("[INFO] Base model loaded (yolov8n-cls.pt)")

    # --- Callbacks for per-epoch progress ---

    batches_per_epoch = max(1, total_images // args.batch)
    _bar: list = [None]
    _epoch_start_t: list = [time.time()]

    def on_train_epoch_start(trainer) -> None:  # type: ignore[no-untyped-def]
        _epoch_start_t[0] = time.time()
        _bar[0] = tqdm(
            total=batches_per_epoch,
            desc=f"Epoch {trainer.epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
        )

    def on_train_batch_end(_trainer) -> None:  # type: ignore[no-untyped-def]
        if _bar[0] is not None:
            _bar[0].update(1)

    def on_fit_epoch_end(trainer) -> None:  # type: ignore[no-untyped-def]
        if _bar[0] is not None:
            _bar[0].close()
            _bar[0] = None
        epoch = trainer.epoch + 1
        elapsed = time.time() - _epoch_start_t[0]
        metrics = trainer.metrics
        top1 = getattr(metrics, "top1", 0.0)
        top5 = getattr(metrics, "top5", 0.0)
        loss = (
            float(trainer.tloss)
            if hasattr(trainer, "tloss") and trainer.tloss is not None
            else 0.0
        )
        print(
            f"[Epoch {epoch:2d}/{args.epochs}]"
            f"  loss={loss:.4f}"
            f"  top1={top1:.3f}  top5={top5:.3f}"
            f"  ({elapsed:.1f}s)"
        )

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    print("[INFO] Starting training ...")
    t0 = time.time()

    output_path = Path(args.output).resolve()
    project_dir = output_path.parent
    run_name = output_path.name

    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    elapsed_total = time.time() - t0
    best = Path(args.output) / "weights" / "best.pt"
    dest = Path("data/models/hand_classifier/hand_classifier.pt")
    print(f"\n[DONE] Total time: {elapsed_total:.0f}s")
    if best.exists():
        import shutil  # pylint: disable=import-outside-toplevel

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, dest)
        print(f"[DONE] Weights copied to {dest}")
    else:
        print(f"[WARN] Expected weights at {best} — check output dir")


if __name__ == "__main__":
    main()

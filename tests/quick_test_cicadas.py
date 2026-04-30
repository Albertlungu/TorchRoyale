"""
Quick test for Cicadas model that outputs annotated images.

Similar to tests/test_full_pipeline.py but focused on the Cicadas model.
Usage:
  python scripts/quick_test_cicadas.py --count 10
  python scripts/quick_test_cicadas.py --video path/to/video.mp4 --count 20 --output-dir output/cicadas_test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parents[1]))


def _detect_device() -> str:
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


def draw_boxes(frame, res) -> int:
    """Draw boxes from ultralytics result on frame and return count."""
    import cv2

    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return 0

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
    except Exception:
        # Fallback: boxes may be list-like or already numpy
        try:
            xyxy = boxes.xyxy.numpy()
            confs = boxes.conf.numpy()
            cls_ids = boxes.cls.numpy()
        except Exception:
            # If we can't extract, just return count
            return len(boxes)

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{int(cls)} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return len(xyxy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test Cicadas model that saves annotated frames")
    parser.add_argument("--video", default=None, help="Optional video file to run on")
    parser.add_argument("--count", type=int, default=10, help="How many frames to run")
    parser.add_argument("--start", type=int, default=0, help="Start frame for video")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for video")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cuda/mps/cpu)")
    parser.add_argument("--weights", default="data/models/onfield/cicadas_best.pt", help="Path to cicadas weights")
    parser.add_argument("--output-dir", default="output/quick_cicadas", help="Output directory for annotated frames")

    args = parser.parse_args()

    device = args.device if args.device != "auto" else _detect_device()
    print(f"Using device: {device}")

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"ultralytics not installed or failed to import: {e}")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Warning: weights not found at {weights_path}. Loading may fail.")

    print("Loading Cicadas model...")
    model = YOLO(str(weights_path))
    try:
        model.to(device)
    except Exception:
        pass

    import cv2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: could not open video {args.video}")
            cap = None
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video opened: {w}x{h}, {total} frames")
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    synthetic_frame = None
    if cap is None:
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print(f"Running {args.count} inferences and saving annotated frames to {output_dir}...")
    ran = 0
    frame_num = args.start
    start_ts = time.time()

    while ran < args.count:
        # handle video stride
        if cap is not None and ran > 0:
            for _ in range(args.stride - 1):
                cap.grab()

        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
        else:
            frame = synthetic_frame.copy()

        # run inference
        try:
            results = model(frame, conf=0.25, max_det=100)
            res0 = results[0] if isinstance(results, (list, tuple)) else results
            n_boxes = draw_boxes(frame, res0)
            info_text = f"Frame {frame_num} | Detections: {n_boxes}"
            cv2.putText(frame, info_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out_path = output_dir / f"frame_{frame_num}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved: {out_path} ({n_boxes} detections)")
        except Exception as e:
            print(f"Inference failed on frame {frame_num}: {e}")

        ran += 1
        frame_num += args.stride
        time.sleep(0.05)

    duration = time.time() - start_ts
    print(f"Completed {ran} inferences in {duration:.2f}s")

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()

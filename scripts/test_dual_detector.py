"""
Smoke test for the DualModelDetector.

Verifies that both models load correctly and the detector runs without
crashing on a blank frame. Zero detections on a blank frame is expected.

Usage:
  python scripts/test_dual_detector.py
"""
from __future__ import annotations

import cv2
import numpy as np

# Ensure project root is on path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.detection.dual_model_detector import DualModelDetector


def main() -> None:
    """Run the smoke test."""
    print("=" * 60)
    print("DualModelDetector Smoke Test")
    print("=" * 60)

    try:
        print("Initializing DualModelDetector...")
        det = DualModelDetector()
        print("Detector initialized successfully.")
    except FileNotFoundError as e:
        print(f"Model weights not found: {e}")
        print("Please run the training scripts first:")
        print("  python scripts/train_cicadas.py")
        print("  python scripts/train_visionbot.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)

    print("\nRunning detection on a blank frame...")
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    try:
        result = det.detect(blank)
        print(f"on_field detections: {len(result.on_field)}")
        print("Smoke test passed -- both models loaded and ran without error.")
    except Exception as e:
        print(f"Error during detection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

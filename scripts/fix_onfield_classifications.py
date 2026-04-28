#!/usr/bin/env python3
"""
Fix Roboflow detection class names where on-field cards are missing '-on-field' suffix.

Roboflow outputs 'hog-rider' instead of 'hog-rider-on-field'. This script patches
the JSON files so is_on_field is set correctly based on class name patterns.
"""

import argparse
import glob
import json
from pathlib import Path

IN_HAND_KEYWORDS = ["in-hand", "in_hand", "inhand"]
NEXT_CARD_KEYWORDS = ["next-card", "next_card", "nextcard", "-next"]


def is_card_in_hand(class_name: str) -> bool:
    class_lower = class_name.lower()
    return any(kw in class_lower for kw in IN_HAND_KEYWORDS)


def is_next_card(class_name: str) -> bool:
    class_lower = class_name.lower()
    return any(kw in class_lower for kw in NEXT_CARD_KEYWORDS)


def is_on_field_card(class_name: str) -> bool:
    if is_card_in_hand(class_name):
        return False
    if is_next_card(class_name):
        return False
    return True


def patch_detection(det: dict) -> dict:
    det = dict(det)
    class_name = det.get("class_name", "")

    if is_on_field_card(class_name):
        det["is_on_field"] = True
    else:
        det["is_on_field"] = False

    return det


def patch_frame(frame: dict) -> dict:
    frame = dict(frame)
    detections = frame.get("detections", [])
    frame["detections"] = [patch_detection(d) for d in detections]
    return frame


def patch_json_file(json_path: Path, dry_run: bool = False) -> tuple[int, int]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
            if not text or not text.strip():
                print(f"Warning: {json_path} is empty; skipping.")
                return 0, 0
            data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: could not parse {json_path}: {e}; skipping.")
        return 0, 0

    patched_frames = 0
    patched_detections = 0

    frames = data.get("frames", [])
    for frame in frames:
        detections = frame.get("detections", [])
        for det in detections:
            original = det.get("is_on_field", False)
            patched = is_on_field_card(det.get("class_name", ""))

            if original != patched:
                det["is_on_field"] = patched
                patched_detections += 1
                patched_frames += 1

    if not dry_run:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return patched_frames, patched_detections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch is_on_field in analysis JSON files"
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        help="JSON files or glob patterns",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    resolved = set()
    for pattern in args.patterns:
        for p in glob.glob(pattern):
            resolved.add(Path(p).resolve())

    if not resolved:
        print("No JSON files found.")
        return 1

    total_frames = 0
    total_detections = 0

    for json_path in sorted(resolved):
        frames, detections = patch_json_file(json_path, dry_run=args.dry_run)
        if detections > 0:
            status = "DRY-RUN" if args.dry_run else "PATCHED"
            print(
                f"[{status}] {json_path.name}: {detections} detections in {frames} frames"
            )
        total_frames += frames
        total_detections += detections

    if total_detections == 0:
        print("No detections needed patching.")
    else:
        action = "Would patch" if args.dry_run else "Patched"
        print(f"\n{action} {total_detections} detections across {total_frames} frames.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

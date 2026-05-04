#!/usr/bin/env python3
"""
Re-run HandTracker on existing analysis JSONs to stabilise hand_cards.

Replaces the hand_cards field in each frame with the tracker's output,
which forward-fills from previous detections and removes played cards.

Usage:
  python scripts/patch_hand_cards.py --analyses-dir output/analysis
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.detection.hand_tracker import HandTracker


def atomic_write(path: Path, data: dict) -> None:
    """
    Write data as JSON to path atomically via a temporary file.

    Args:
        path: destination path.
        data: JSON-serialisable dict.
    """
    tmp = NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8"
    )
    try:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        tmp.close()
        shutil.move(tmp.name, str(path))
    finally:
        try:
            Path(tmp.name).unlink()
        except Exception:  # pylint: disable=broad-exception-caught
            pass


def process(path: Path, dry_run: bool) -> None:
    """
    Re-track hand cards for all frames in a single analysis JSON.

    Args:
        path:    path to the *_analysis.json file.
        dry_run: if True, write output to a 'dryrun' subdirectory instead.
    """
    print(f"Processing {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "frames" not in data:
        print("  Skipping: no frames")
        return

    tracker = HandTracker()
    changed = False
    frames: List[dict] = data["frames"]
    frame_count = len(frames)

    for idx, fr in enumerate(frames):
        tracked: List[str] = tracker.update(fr.get("detections", []))
        if fr.get("hand_cards") != tracked:
            fr["hand_cards"] = tracked
            changed = True
        print(
            f"  [{idx+1}/{frame_count}] ts={fr.get('timestamp_ms')}ms  hand={tracked}",
            flush=True,
        )

    if not changed:
        print("  No changes")
        return

    if dry_run:
        dry_dir = path.parent / "dryrun"
        dry_dir.mkdir(exist_ok=True)
        atomic_write(dry_dir / path.name, data)
        print(f"  Dry-run: {dry_dir / path.name}")
    else:
        atomic_write(path, data)
        print(f"  Updated {path.name}")


def main() -> None:
    """Parse arguments and re-track hand cards for all matching analysis files."""
    parser = argparse.ArgumentParser(
        description="Re-run HandTracker on existing analysis JSONs to stabilise hand_cards."
    )
    parser.add_argument("--analyses-dir", default="output/analysis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(Path(args.analyses_dir).rglob("*_analysis.json"))
    if not files:
        print("No analysis files found")
        return
    print(f"Processing {len(files)} files")
    for analysis_file in files:
        process(analysis_file, args.dry_run)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Patch hand_cards in existing analysis JSONs using HandTracker.

Replays the detections already stored in each JSON through the
HandTracker to produce a stable 4-card hand state, then writes the
corrected hand_cards back to the JSON atomically.

Usage:
  python3 scripts/patch_hand_cards.py --analyses-dir output/analysis
"""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import sys
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.game_state.hand_tracker import HandTracker


def atomic_write_json(path: Path, data: dict):
    tmp = NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    try:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        tmp.close()
        shutil.move(tmp.name, str(path))
    finally:
        try:
            Path(tmp.name).unlink()
        except Exception:
            pass


def process_file(path: Path, dry_run: bool = False) -> bool:
    print(f"Processing {path.name}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  Failed to read: {e}")
        return False

    if "frames" not in data:
        print(f"  Skipping: no frames")
        return False

    tracker = HandTracker()
    changed = False
    for fr in data["frames"]:
        tracked = tracker.update(fr.get("detections", []))
        if fr.get("hand_cards") != tracked:
            fr["hand_cards"] = tracked
            changed = True

    if not changed:
        print(f"  No changes")
        return True

    if dry_run:
        dry_dir = path.parent / "dryrun"
        dry_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(dry_dir / path.name, data)
        print(f"  Dry-run: wrote {dry_dir / path.name}")
    else:
        atomic_write_json(path, data)
        print(f"  Updated {path.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses-dir", default="output/analysis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    analyses_dir = Path(args.analyses_dir)
    if not analyses_dir.exists():
        print(f"Directory not found: {analyses_dir}")
        raise SystemExit(2)

    files = sorted(p for p in analyses_dir.rglob("*_analysis.json") if p.is_file())
    if not files:
        print("No analysis files found")
        return

    print(f"Processing {len(files)} files")
    for f in files:
        process_file(f, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

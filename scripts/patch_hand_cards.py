#!/usr/bin/env python3
"""
Re-run HandTracker on existing analysis JSONs to stabilise hand_cards.

Usage:
  python scripts/patch_hand_cards.py --analyses-dir output/analysis
"""
import argparse, json, shutil, sys
from pathlib import Path
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.detection.hand_tracker import HandTracker


def atomic_write(path: Path, data: dict):
    tmp = NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp")
    try:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush(); tmp.close()
        shutil.move(tmp.name, str(path))
    finally:
        try: Path(tmp.name).unlink()
        except: pass


def process(path: Path, dry_run: bool):
    print(f"Processing {path.name}")
    data = json.loads(path.read_text())
    if "frames" not in data:
        print("  Skipping: no frames"); return

    tracker = HandTracker()
    changed = False
    frames = data["frames"]
    n = len(frames)

    for i, fr in enumerate(frames):
        tracked = tracker.update(fr.get("detections", []))
        if fr.get("hand_cards") != tracked:
            fr["hand_cards"] = tracked
            changed = True
        print(f"  [{i+1}/{n}] ts={fr.get('timestamp_ms')}ms  hand={tracked}", flush=True)

    if not changed:
        print("  No changes"); return

    if dry_run:
        d = path.parent / "dryrun"; d.mkdir(exist_ok=True)
        atomic_write(d / path.name, data)
        print(f"  Dry-run: {d / path.name}")
    else:
        atomic_write(path, data)
        print(f"  Updated {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses-dir", default="output/analysis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(Path(args.analyses_dir).rglob("*_analysis.json"))
    if not files:
        print("No analysis files found"); return
    print(f"Processing {len(files)} files")
    for f in files:
        process(f, args.dry_run)


if __name__ == "__main__":
    main()

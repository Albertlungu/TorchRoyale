#!/usr/bin/env python3
"""
Patch player_elixir in analysis JSONs by reading the elixir number via OCR.

Usage:
  python3 scripts/patch_player_elixir_from_video.py --analyses-dir output

The script:
 - Finds all files matching "*_analysis.json" under the given directory
 - For each JSON, opens the referenced video and seeks to each frame's
   `timestamp_ms` to read the elixir number from the UI
 - Uses `UIRegions.elixir_number` and `DigitDetector.detect_elixir()` to
   read the on-screen elixir count (0-10)
 - Fills `player_elixir` for frames where detection succeeded
 - Fills remaining frames using nearest-neighbour interpolation
 - Writes the updated JSON atomically (safe replace)

Requirements: opencv-python, numpy; EasyOCR needed if `--preload-ocr` is used
"""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

try:
    import cv2
    import numpy as np
except Exception as e:
    print(f"Missing dependency: {e}")
    raise SystemExit(2)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants.ui_regions import UIRegions
from src.ocr.digit_detector import DigitDetector


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


def fill_nearest(detected: List[Optional[int]]) -> List[Optional[int]]:
    """Fill None entries using the nearest detected value (neighbour interpolation)."""
    n = len(detected)
    out = detected.copy()
    known = [i for i, v in enumerate(detected) if v is not None]
    if not known:
        return out
    for idx in range(n):
        if out[idx] is not None:
            continue
        prev = max((k for k in known if k < idx), default=None)
        nxt = min((k for k in known if k > idx), default=None)
        if prev is None:
            out[idx] = out[nxt]
        elif nxt is None:
            out[idx] = out[prev]
        else:
            # pick whichever anchor is closer
            out[idx] = out[prev] if (idx - prev) <= (nxt - idx) else out[nxt]
    return out


def process_file(
    path: Path,
    preload_ocr: bool = False,
    dry_run: bool = False,
    override_video: Optional[Path] = None,
) -> bool:
    print(f"Processing {path}")
    try:
        text = path.read_text()
        if not text or not text.strip():
            print(f"Skipping empty JSON file: {path}")
            return False
        data = json.loads(text)
    except Exception as e:
        print(f"Failed to read JSON {path}: {e}")
        return False

    if "video_info" not in data or "frames" not in data:
        print(f"Skipping {path}: missing video_info or frames")
        return False

    if override_video is not None:
        video_path = Path(override_video)
    else:
        video_path = Path(data["video_info"].get("path", ""))
        if not video_path.is_absolute():
            video_path = (PROJECT_ROOT / video_path).resolve()

    if not video_path.exists():
        replays_dir = PROJECT_ROOT / "data" / "replays"
        found = None
        if replays_dir.exists():
            stem = video_path.stem.lower()
            candidates = []
            for p in replays_dir.rglob("*"):
                if not p.is_file():
                    continue
                pn = p.stem.lower()
                if pn == stem:
                    found = p
                    break
                if stem in pn:
                    candidates.append(p)
            if found is None and candidates:
                candidates.sort(key=lambda x: len(x.name))
                found = candidates[0]

        if found is not None:
            print(f"Resolved video path by search: {found}")
            video_path = found
        else:
            print(f"Video not found: {video_path}")
            return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False

    vid_w = int(data["video_info"].get("width", 0))
    vid_h = int(data["video_info"].get("height", 0))
    if vid_w == 0 or vid_h == 0:
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read any frames from {video_path}")
            cap.release()
            return False
        vid_h, vid_w = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ui = UIRegions(vid_w, vid_h)
    detector = DigitDetector(preload_ocr=preload_ocr)

    timestamps = [int(f.get("timestamp_ms", 0)) for f in data["frames"]]
    detected: List[Optional[int]] = [None] * len(data["frames"])

    for i, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_ui = UIRegions(vid_w, vid_h)
        frame_ui.align_elixir_to_image(frame)
        result = detector.detect_elixir(frame, frame_ui.elixir_number.to_tuple())
        if result.detected:
            detected[i] = int(result.value)
            print(f"  frame {i} ts={ts}ms -> detected elixir={result.value}")

    cap.release()

    filled = fill_nearest(detected)

    n_detected = sum(1 for v in detected if v is not None)
    print(f"  Detected {n_detected}/{len(timestamps)} frames via OCR")

    changed = False
    for i, val in enumerate(filled):
        if val is None:
            continue
        if data["frames"][i].get("player_elixir") != val:
            data["frames"][i]["player_elixir"] = val
            changed = True

    if changed:
        if dry_run:
            dry_dir = path.parent / "dryrun"
            dry_dir.mkdir(parents=True, exist_ok=True)
            dry_path = dry_dir / path.name
            atomic_write_json(dry_path, data)
            print(f"Dry-run: wrote {dry_path}")
        else:
            atomic_write_json(path, data)
            print(f"Updated {path}")
    else:
        print(f"No updates for {path}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyses-dir",
        default="output",
        help="Directory to search for *_analysis.json",
    )
    parser.add_argument(
        "--videos-dir",
        default="data/replays",
        help="Directory containing replay videos",
    )
    parser.add_argument(
        "--pairing",
        choices=["name", "order"],
        default="name",
        help="How to pair analyses with videos: by name or by sorted order",
    )
    parser.add_argument(
        "--preload-ocr",
        action="store_true",
        help="Preload EasyOCR model before processing (faster if running many files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not overwrite originals; write to dryrun/ subfolders instead",
    )
    args = parser.parse_args()

    analyses_base = Path(args.analyses_dir)
    videos_base = Path(args.videos_dir)
    if not analyses_base.exists():
        print(f"Analyses directory not found: {analyses_base}")
        raise SystemExit(2)
    if not videos_base.exists():
        print(f"Videos directory not found: {videos_base}")
        raise SystemExit(2)

    analyses = sorted(
        [p for p in analyses_base.rglob("*_analysis.json") if p.is_file()]
    )
    videos = sorted([p for p in videos_base.rglob("*") if p.is_file()])

    if not analyses:
        print(f"No *_analysis.json files found under {analyses_base}")
        return 0

    pairs = []
    if args.pairing == "order":
        count = min(len(analyses), len(videos))
        if len(analyses) != len(videos):
            print(
                f"Warning: different counts analyses={len(analyses)} videos={len(videos)}; "
                f"pairing by order using first {count} items"
            )
        for i in range(count):
            pairs.append((analyses[i], videos[i]))
    else:
        video_map = {v.stem.lower(): v for v in videos}
        for a in analyses:
            a_name = a.stem
            base_name = a_name[:-9] if a_name.endswith("_analysis") else a_name
            found = video_map.get(base_name.lower())
            if not found:
                for vstem, vp in video_map.items():
                    if base_name.lower() in vstem:
                        found = vp
                        break
            if found:
                pairs.append((a, found))
            else:
                print(
                    f"No matching video found for {a.name}; "
                    f"will use path inside JSON if present"
                )
                pairs.append((a, None))

    print(f"Processing {len(pairs)} analysis->video pairs (pairing={args.pairing})")
    for analysis_path, video_path in pairs:
        label = video_path.name if video_path else "(use JSON video path)"
        print(f"Pair: {analysis_path.name}  <--  {label}")
        process_file(
            analysis_path,
            preload_ocr=args.preload_ocr,
            dry_run=args.dry_run,
            override_video=video_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Patch game_time_remaining in analysis JSONs by reading the video timer via OCR.

Usage:
  python3 scripts/patch_game_time_from_video.py --dir output/replay_runs/analysis

The script:
 - Finds all files matching "*_analysis.json" under the given directory
 - For each JSON, opens the referenced video and seeks to each frame's
   `timestamp_ms` to read the frame containing the timer
 - Uses `UIRegions.align_timer_to_image()` and `DigitDetector.detect_timer()`
   to read the on-screen timer (seconds remaining)
 - Fills `game_time_remaining` for frames where detection succeeded
 - Interpolates/extrapolates missing values using nearby detections
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


def interpolate_game_times(
    timestamps_ms: List[int], detected_secs: List[Optional[int]]
) -> List[Optional[int]]:
    """
    Given parallel lists of timestamps (ms) and detected seconds (or None),
    fill missing values by nearest interpolation/extrapolation assuming the
    timer counts down in real time (1 sec per 1000 ms).
    """
    n = len(timestamps_ms)
    out = detected_secs.copy()

    # Collect indices with known values
    known = [i for i, v in enumerate(detected_secs) if v is not None]
    if not known:
        return out

    # Fill forward/backward from known points
    for idx in range(n):
        if out[idx] is not None:
            continue
        # find nearest known before and after
        prev = max((k for k in known if k < idx), default=None)
        nxt = min((k for k in known if k > idx), default=None)

        if prev is None and nxt is None:
            out[idx] = None
        elif prev is None:
            # extrapolate backward from nxt
            dt_ms = timestamps_ms[nxt] - timestamps_ms[idx]
            secs = detected_secs[nxt] + int(round(dt_ms / 1000.0))
            out[idx] = max(0, secs)
        elif nxt is None:
            # extrapolate forward from prev
            dt_ms = timestamps_ms[idx] - timestamps_ms[prev]
            secs = detected_secs[prev] - int(round(dt_ms / 1000.0))
            out[idx] = max(0, secs)
        else:
            # Interpolate between prev and nxt linearly (but timer should
            # decrease by roughly 1/sec). We'll anchor to prev value.
            dt_ms = timestamps_ms[idx] - timestamps_ms[prev]
            secs = detected_secs[prev] - int(round(dt_ms / 1000.0))
            out[idx] = max(0, secs)

    return out


def _is_anomalous(secs: int, prev_detected: Optional[int]) -> bool:
    """Return True if the detected value looks like a bad OCR read."""
    if secs > 200:
        return True
    if prev_detected is not None and secs > prev_detected + 30:
        return True
    return False


def _save_debug_frame(
    frame: np.ndarray,
    timer_region: tuple,
    secs: int,
    frame_idx: int,
    ts: int,
    debug_dir: Path,
    detector,
) -> None:
    """Save the full frame and the preprocessed timer ROI for inspection."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame{frame_idx:05d}_ts{ts}ms_detected{secs}s"

    # Full frame with timer ROI highlighted
    annotated = frame.copy()
    x1, y1, x2, y2 = timer_region
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        annotated,
        f"detected={secs}s",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(str(debug_dir / f"{stem}_full.jpg"), annotated)

    # Raw ROI crop
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        cv2.imwrite(str(debug_dir / f"{stem}_roi_raw.jpg"), roi)

    # Preprocessed ROI (what OCR actually sees)
    processed = detector._preprocess_for_ocr(roi)
    if processed is not None and processed.size > 0:
        cv2.imwrite(str(debug_dir / f"{stem}_roi_ocr.jpg"), processed)


def process_file(
    path: Path,
    preload_ocr: bool = False,
    dry_run: bool = False,
    override_video: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
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

    # If an override video path was provided via pairing, use it.
    if override_video is not None:
        video_path = Path(override_video)
    else:
        video_path = Path(data["video_info"].get("path", ""))
        if not video_path.is_absolute():
            video_path = (PROJECT_ROOT / video_path).resolve()

    if not video_path.exists():
        # Try to resolve by searching data/replays for a matching stem (case-insensitive)
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
                # choose the shortest candidate name (likely the closest match)
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

    # Use the video_info width/height if present, otherwise derived from first frame
    vid_w = int(data["video_info"].get("width", 0))
    vid_h = int(data["video_info"].get("height", 0))
    if vid_w == 0 or vid_h == 0:
        # read one frame to get size
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
    detected = [None] * len(data["frames"])
    last_detected: Optional[int] = None
    frames_seen = 0

    for i, ts in enumerate(timestamps):
        # Seek by milliseconds (more precise than frame number when video fps varies)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ret, frame = cap.read()
        if not ret or frame is None:
            # skip if frame not available
            continue

        # Use a fresh UIRegions per frame to avoid carry-over side-effects
        frame_ui = UIRegions(vid_w, vid_h)
        align = frame_ui.align_timer_to_image(frame)
        if align is not None:
            # align_timer_to_image already updated frame_ui.timer
            pass

        secs = detector.detect_timer(frame, frame_ui.timer.to_tuple())

        if debug_dir is not None:
            save_this = frames_seen < 3 or (secs is not None and _is_anomalous(secs, last_detected))
            if save_this:
                label = secs if secs is not None else "None"
                file_debug_dir = debug_dir / path.stem
                _save_debug_frame(
                    frame,
                    frame_ui.timer.to_tuple(),
                    label,
                    i,
                    ts,
                    file_debug_dir,
                    detector,
                )
                tag = "preview" if frames_seen < 3 else "anomalous"
                print(f"    [DEBUG:{tag}] frame {i} saved to {file_debug_dir}")

        frames_seen += 1

        if secs is not None:
            detected[i] = int(secs)
            print(f"  frame {i} ts={ts}ms -> detected {secs}s")
            last_detected = secs

    # Interpolate/extrapolate missing values
    filled = interpolate_game_times(timestamps, detected)

    # Apply back to data
    changed = False
    for i, val in enumerate(filled):
        if val is None:
            continue
        if data["frames"][i].get("game_time_remaining") != val:
            data["frames"][i]["game_time_remaining"] = val
            changed = True

    if changed:
        if dry_run:
            # Recreate the file under a dryrun/ subdirectory next to original
            dry_dir = path.parent / "dryrun"
            dry_dir.mkdir(parents=True, exist_ok=True)
            dry_path = dry_dir / path.name
            atomic_write_json(dry_path, data)
            print(f"Dry-run: recreated {dry_path}")
        else:
            atomic_write_json(path, data)
            print(f"Updated {path}")
    else:
        print(f"No updates for {path}")

    cap.release()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyses-dir",
        default="output/replay_runs/analysis",
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
        help="Preload EasyOCR model (requires easyocr installed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not overwrite originals; recreate under dryrun/ subfolders",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="If set, save annotated frames for anomalous timer reads to this directory",
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
                f"Warning: different counts analyses={len(analyses)} videos={len(videos)}; pairing by order using first {count} items"
            )
        for i in range(count):
            pairs.append((analyses[i], videos[i]))
    else:
        # name pairing: match analysis stem without trailing _analysis to video stems
        video_map = {v.stem.lower(): v for v in videos}
        for a in analyses:
            a_name = a.stem
            if a_name.endswith("_analysis"):
                base_name = a_name[:-9]
            else:
                base_name = a_name
            # try exact stem match
            found = video_map.get(base_name.lower())
            if not found:
                # try substring match
                for vstem, vp in video_map.items():
                    if base_name.lower() in vstem:
                        found = vp
                        break
            if found:
                pairs.append((a, found))
            else:
                print(
                    f"No matching video found for analysis {a.name}; it will be processed using video path inside JSON if present"
                )
                pairs.append((a, None))

    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    print(f"Processing {len(pairs)} analysis->video pairs (pairing={args.pairing})")
    for analysis_path, video_path in pairs:
        if video_path is not None:
            print(f"Pair: {analysis_path.name}  <--  {video_path.name}")
        else:
            print(f"Pair: {analysis_path.name}  <--  (use JSON video path)")
        # process_file will resolve video path from JSON if override is None
        if video_path is not None:
            process_file(
                analysis_path,
                preload_ocr=args.preload_ocr,
                dry_run=args.dry_run,
                override_video=video_path,
                debug_dir=debug_dir,
            )
        else:
            process_file(
                analysis_path,
                preload_ocr=args.preload_ocr,
                dry_run=args.dry_run,
                debug_dir=debug_dir,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

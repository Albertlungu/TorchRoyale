#!/usr/bin/env python3
"""
Patch timer, elixir, multiplier, and game_phase in existing analysis JSONs
by re-reading the video with OCR. Performs a single video pass per file.

Usage:
  python scripts/patch_ocr_fields.py --analyses-dir output/analysis --preload-ocr
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2

from src.game_state.phase import derive_phases
from src.ocr.detector import DigitDetector
from src.ocr.regions import UIRegions


def atomic_write(path: Path, data: dict) -> None:
    """
    Write data as JSON to path atomically via a temporary file.

    Args:
        path: destination path.
        data: JSON-serialisable dict.
    """
    tmp = NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8")
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


def interpolate_timer(
    timestamps: List[int], values: List[Optional[int]]
) -> List[Optional[int]]:
    """
    Fill None timer values by linear interpolation from neighbouring known values.

    Args:
        timestamps: per-frame timestamps in milliseconds.
        values:     per-frame timer values (seconds); None where OCR failed.

    Returns:
        New list with None gaps filled by extrapolation from the nearest known value.
    """
    count = len(timestamps)
    out = values.copy()
    known = [idx for idx, val in enumerate(values) if val is not None]
    if not known:
        return out
    for frame_idx in range(count):
        if out[frame_idx] is not None:
            continue
        prev = max((k for k in known if k < frame_idx), default=None)
        nxt  = min((k for k in known if k > frame_idx), default=None)
        if prev is None and nxt is None:
            continue
        if prev is None:
            out[frame_idx] = max(
                0, values[nxt] + int(round((timestamps[nxt] - timestamps[frame_idx]) / 1000))
            )
        else:
            out[frame_idx] = max(
                0,
                values[prev] - int(round((timestamps[frame_idx] - timestamps[prev]) / 1000)),
            )
    return out


def fill_nearest(values: List[Optional[int]]) -> List[Optional[int]]:
    """
    Fill None values with the nearest known neighbour (prefer left tie-break).

    Args:
        values: list with possible None gaps.

    Returns:
        New list with None gaps filled.
    """
    count = len(values)
    out = values.copy()
    known = [idx for idx, val in enumerate(values) if val is not None]
    if not known:
        return out
    for frame_idx in range(count):
        if out[frame_idx] is not None:
            continue
        prev = max((k for k in known if k < frame_idx), default=None)
        nxt  = min((k for k in known if k > frame_idx), default=None)
        if prev is None:
            out[frame_idx] = out[nxt]
        elif nxt is None:
            out[frame_idx] = out[prev]
        else:
            out[frame_idx] = out[prev] if (frame_idx - prev) <= (nxt - frame_idx) else out[nxt]
    return out


def process(path: Path, video_path: Path, preload: bool, dry_run: bool) -> None:
    """
    Re-run OCR on a single analysis JSON and update timer/elixir/phase fields.

    Args:
        path:       path to the *_analysis.json file.
        video_path: path to the corresponding replay video.
        preload:    if True, warm up the EasyOCR reader before processing.
        dry_run:    if True, write patched JSON to a 'dryrun' subdirectory instead.
    """
    print(f"Processing {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "frames" not in data or "video_info" not in data:
        print("  Skipping: bad format")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return

    vid_w = int(data["video_info"].get("width", 0)) or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(data["video_info"].get("height", 0)) or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = DigitDetector(preload=preload)
    frames = data["frames"]
    frame_count = len(frames)
    ts_list: List[int] = [int(fr.get("timestamp_ms", 0)) for fr in frames]

    timer_det:   List[Optional[int]] = [None] * frame_count
    elixir_det:  List[Optional[int]] = [None] * frame_count
    mult_det:    List[Optional[int]] = [None] * frame_count

    for i, ts in enumerate(ts_list):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        ui = UIRegions(vid_w, vid_h)
        ui.align_timer(frame)
        ui.align_elixir(frame)
        ui.align_multiplier(frame)

        secs = detector.detect_timer(frame, ui.timer.to_tuple())
        if secs is not None:
            timer_det[i] = secs

        elixir_result = detector.detect_elixir(frame, ui.elixir_number.to_tuple())
        if elixir_result.detected:
            elixir_det[i] = int(elixir_result.value)

        mult_det[i] = detector.detect_multiplier(frame, ui.multiplier_icon.to_tuple())

        elixir_display = elixir_result.value if elixir_result.detected else "?"
        print(
            f"  [{i+1}/{frame_count}] ts={ts}ms  timer={secs}"
            f"  elixir={elixir_display}  x{mult_det[i]}",
            flush=True,
        )

    cap.release()

    first_timer = next((idx for idx, val in enumerate(timer_det) if val is not None), None)
    if first_timer is None:
        print("  No timer detections — skipping")
        return
    if first_timer > 0:
        print(f"  Trimming {first_timer} pre-game frames")
        frames = frames[first_timer:]
        timer_det  = timer_det[first_timer:]
        elixir_det = elixir_det[first_timer:]
        mult_det   = mult_det[first_timer:]
        ts_list    = ts_list[first_timer:]
        data["frames"] = frames
        frame_count = len(frames)

    timer_filled  = interpolate_timer(ts_list, timer_det)
    elixir_filled = fill_nearest(elixir_det)
    multipliers, phases = derive_phases(timer_filled)

    # Blend OCR multiplier when it detects x2 or x3
    for idx, mult in enumerate(mult_det):
        if mult is not None and mult > 1:
            multipliers[idx] = mult

    changed = False
    for idx in range(frame_count):
        fr = frames[idx]
        if timer_filled[idx] is not None and fr.get("game_time_remaining") != timer_filled[idx]:
            fr["game_time_remaining"] = timer_filled[idx]
            changed = True
        if elixir_filled[idx] is not None and fr.get("player_elixir") != elixir_filled[idx]:
            fr["player_elixir"] = elixir_filled[idx]
            changed = True
        if fr.get("elixir_multiplier") != multipliers[idx]:
            fr["elixir_multiplier"] = multipliers[idx]
            changed = True
        if fr.get("game_phase") != phases[idx]:
            fr["game_phase"] = phases[idx]
            changed = True

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
    """Parse arguments and patch OCR fields in all matching analysis JSONs."""
    parser = argparse.ArgumentParser(
        description="Re-run OCR on existing analysis JSONs to patch timer/elixir/phase fields."
    )
    parser.add_argument("--analyses-dir", default="output/analysis")
    parser.add_argument("--videos-dir", default="data/replays")
    parser.add_argument("--pairing", choices=["name", "order"], default="name")
    parser.add_argument("--preload-ocr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    analyses = sorted(Path(args.analyses_dir).rglob("*_analysis.json"))
    videos = sorted(Path(args.videos_dir).rglob("*"))
    video_map = {vid.stem.lower(): vid for vid in videos if vid.is_file()}

    for analysis in analyses:
        base = analysis.stem[:-9] if analysis.stem.endswith("_analysis") else analysis.stem
        vid = video_map.get(base.lower())
        if vid is None:
            vid = next((v for k, v in video_map.items() if base.lower() in k), None)
        if vid is None:
            print(f"No video for {analysis.name}")
            continue
        process(analysis, vid, args.preload_ocr, args.dry_run)


if __name__ == "__main__":
    main()

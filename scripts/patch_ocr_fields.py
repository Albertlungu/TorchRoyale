#!/usr/bin/env python3
"""
Patch timer, elixir, multiplier, and game_phase in existing analysis JSONs
by re-reading the video with OCR. Single video pass per file.

Usage:
  python scripts/patch_ocr_fields.py --analyses-dir output/analysis --preload-ocr
"""
import argparse, json, shutil, sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
import numpy as np

from src.ocr.detector import DigitDetector
from src.ocr.regions import UIRegions
from src.game_state.phase import derive_phases


def atomic_write(path: Path, data: dict):
    tmp = NamedTemporaryFile("w", delete=False, dir=str(path.parent), suffix=".tmp")
    try:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush(); tmp.close()
        shutil.move(tmp.name, str(path))
    finally:
        try: Path(tmp.name).unlink()
        except: pass


def interpolate_timer(ts: List[int], vals: List[Optional[int]]) -> List[Optional[int]]:
    n, out = len(ts), vals.copy()
    known = [i for i, v in enumerate(vals) if v is not None]
    if not known: return out
    for idx in range(n):
        if out[idx] is not None: continue
        prev = max((k for k in known if k < idx), default=None)
        nxt  = min((k for k in known if k > idx), default=None)
        if prev is None and nxt is None: continue
        elif prev is None:
            out[idx] = max(0, vals[nxt] + int(round((ts[nxt]-ts[idx])/1000)))
        elif nxt is None:
            out[idx] = max(0, vals[prev] - int(round((ts[idx]-ts[prev])/1000)))
        else:
            out[idx] = max(0, vals[prev] - int(round((ts[idx]-ts[prev])/1000)))
    return out


def fill_nearest(vals: List[Optional[int]]) -> List[Optional[int]]:
    n, out = len(vals), vals.copy()
    known = [i for i, v in enumerate(vals) if v is not None]
    if not known: return out
    for idx in range(n):
        if out[idx] is not None: continue
        prev = max((k for k in known if k < idx), default=None)
        nxt  = min((k for k in known if k > idx), default=None)
        if prev is None:   out[idx] = out[nxt]
        elif nxt is None:  out[idx] = out[prev]
        else: out[idx] = out[prev] if (idx-prev) <= (nxt-idx) else out[nxt]
    return out


def process(path: Path, video_path: Path, preload: bool, dry_run: bool):
    print(f"Processing {path.name}")
    data = json.loads(path.read_text())
    if "frames" not in data or "video_info" not in data:
        print("  Skipping: bad format")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open video: {video_path}")
        return

    vid_w = int(data["video_info"].get("width", 0)) or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(data["video_info"].get("height", 0)) or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = DigitDetector(preload=preload)
    frames = data["frames"]
    n = len(frames)
    ts_list = [int(f.get("timestamp_ms", 0)) for f in frames]

    timer_det:   List[Optional[int]] = [None] * n
    elixir_det:  List[Optional[int]] = [None] * n
    mult_det:    List[Optional[int]] = [None] * n
    prev_timer = None

    for i, ts in enumerate(ts_list):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ret, frame = cap.read()
        if not ret or frame is None: continue

        ui = UIRegions(vid_w, vid_h)
        ui.align_timer(frame); ui.align_elixir(frame); ui.align_multiplier(frame)

        secs = det.detect_timer(frame, ui.timer.to_tuple())
        if secs is not None:
            timer_det[i] = secs
            prev_timer = secs

        er = det.detect_elixir(frame, ui.elixir_number.to_tuple())
        if er.detected:
            elixir_det[i] = int(er.value)

        mult_det[i] = det.detect_multiplier(frame, ui.multiplier_icon.to_tuple())

        print(f"  [{i+1}/{n}] ts={ts}ms  timer={secs}  elixir={er.value if er.detected else '?'}  x{mult_det[i]}", flush=True)

    cap.release()

    first_timer = next((i for i, v in enumerate(timer_det) if v is not None), None)
    if first_timer is None:
        print("  No timer detections — skipping")
        return
    if first_timer > 0:
        print(f"  Trimming {first_timer} pre-game frames")
        frames = frames[first_timer:]
        timer_det = timer_det[first_timer:]
        elixir_det = elixir_det[first_timer:]
        mult_det = mult_det[first_timer:]
        ts_list = ts_list[first_timer:]
        data["frames"] = frames
        n = len(frames)

    timer_filled  = interpolate_timer(ts_list, timer_det)
    elixir_filled = fill_nearest(elixir_det)
    multipliers, phases = derive_phases(timer_filled)

    # Blend OCR multiplier when it sees x2 or x3
    for i, m in enumerate(mult_det):
        if m is not None and m > 1:
            multipliers[i] = m

    changed = False
    for i in range(n):
        fr = frames[i]
        if timer_filled[i] is not None and fr.get("game_time_remaining") != timer_filled[i]:
            fr["game_time_remaining"] = timer_filled[i]; changed = True
        if elixir_filled[i] is not None and fr.get("player_elixir") != elixir_filled[i]:
            fr["player_elixir"] = elixir_filled[i]; changed = True
        if fr.get("elixir_multiplier") != multipliers[i]:
            fr["elixir_multiplier"] = multipliers[i]; changed = True
        if fr.get("game_phase") != phases[i]:
            fr["game_phase"] = phases[i]; changed = True

    if not changed:
        print("  No changes")
        return

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
    parser.add_argument("--videos-dir", default="data/replays")
    parser.add_argument("--pairing", choices=["name", "order"], default="name")
    parser.add_argument("--preload-ocr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    analyses = sorted(Path(args.analyses_dir).rglob("*_analysis.json"))
    videos   = sorted(Path(args.videos_dir).rglob("*"))
    video_map = {v.stem.lower(): v for v in videos if v.is_file()}

    for a in analyses:
        base = a.stem[:-9] if a.stem.endswith("_analysis") else a.stem
        vid = video_map.get(base.lower())
        if vid is None:
            vid = next((v for k, v in video_map.items() if base.lower() in k), None)
        if vid is None:
            print(f"No video for {a.name}"); continue
        process(a, vid, args.preload_ocr, args.dry_run)


if __name__ == "__main__":
    main()

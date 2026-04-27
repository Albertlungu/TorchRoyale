#!/usr/bin/env python3
"""
Patch all OCR-derived fields in analysis JSONs in a single video pass:
  - game_time_remaining  (timer OCR)
  - player_elixir        (elixir number OCR)
  - elixir_multiplier    (derived from patched timer via GamePhaseTracker)
  - game_phase           (derived from patched timer via GamePhaseTracker)

Usage:
  python3 scripts/patch_ocr_fields_from_video.py --analyses-dir output/analysis

Both timer and elixir are detected in the same frame loop so the video is
only read once per file.
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
from src.game_state.game_phase import GamePhaseTracker


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


def interpolate_timer(
    timestamps_ms: List[int], detected: List[Optional[int]]
) -> List[Optional[int]]:
    """Fill missing timer values assuming 1 second per 1000 ms countdown."""
    n = len(timestamps_ms)
    out = detected.copy()
    known = [i for i, v in enumerate(detected) if v is not None]
    if not known:
        return out
    for idx in range(n):
        if out[idx] is not None:
            continue
        prev = max((k for k in known if k < idx), default=None)
        nxt  = min((k for k in known if k > idx), default=None)
        if prev is None and nxt is None:
            out[idx] = None
        elif prev is None:
            dt_ms = timestamps_ms[nxt] - timestamps_ms[idx]
            out[idx] = max(0, detected[nxt] + int(round(dt_ms / 1000.0)))
        elif nxt is None:
            dt_ms = timestamps_ms[idx] - timestamps_ms[prev]
            out[idx] = max(0, detected[prev] - int(round(dt_ms / 1000.0)))
        else:
            dt_ms = timestamps_ms[idx] - timestamps_ms[prev]
            out[idx] = max(0, detected[prev] - int(round(dt_ms / 1000.0)))
    return out


def fill_nearest(detected: List[Optional[int]]) -> List[Optional[int]]:
    """Fill None entries using nearest detected value."""
    n = len(detected)
    out = detected.copy()
    known = [i for i, v in enumerate(detected) if v is not None]
    if not known:
        return out
    for idx in range(n):
        if out[idx] is not None:
            continue
        prev = max((k for k in known if k < idx), default=None)
        nxt  = min((k for k in known if k > idx), default=None)
        if prev is None:
            out[idx] = out[nxt]
        elif nxt is None:
            out[idx] = out[prev]
        else:
            out[idx] = out[prev] if (idx - prev) <= (nxt - idx) else out[nxt]
    return out


def derive_phase_fields(
    timestamps_ms: List[int],
    timer_filled: List[Optional[int]],
    multiplier_detected: Optional[List[Optional[int]]] = None,
) -> tuple:
    """
    Derive elixir_multiplier and game_phase from the filled timer sequence.

    Bypasses GamePhaseTracker to avoid the towers_tied dependency — the
    tracker's overtime detection requires knowing whether towers are tied,
    which we don't have at patch time. Instead we detect overtime directly
    by looking for the timer jumping up from near-zero to >=100 (the 2:00
    overtime reset).

    Regular game (180 -> 0):
      timer > 60  -> single (x1)
      timer <= 60 -> double (x2)
    Overtime (120 -> 0):
      timer > 60  -> sudden_death (x2)
      timer <= 60 -> triple (x3)
    """
    in_overtime = False
    prev_timer: Optional[int] = None
    multipliers: List[int] = []
    phases: List[str] = []

    for i, secs in enumerate(timer_filled):
        # x3 icon detected directly → we're in overtime
        if multiplier_detected and multiplier_detected[i] == 3:
            in_overtime = True
        # Also detect overtime via timer jumping up from near-zero to >=100
        if prev_timer is not None and secs is not None:
            if prev_timer <= 10 and secs >= 100:
                in_overtime = True

        if secs is None:
            phase = phases[-1] if phases else "single"
            mult  = multipliers[-1] if multipliers else 1
        elif secs <= 0:
            phase = "game_over"
            mult  = 1
        elif in_overtime:
            if secs <= 60:
                phase = "triple"
                mult  = 3
            else:
                phase = "sudden_death"
                mult  = 2
        else:
            if secs <= 60:
                phase = "double"
                mult  = 2
            else:
                phase = "single"
                mult  = 1

        phases.append(phase)
        multipliers.append(mult)

        if secs is not None:
            prev_timer = secs

    return multipliers, phases


def _save_debug(frame, label: str, debug_dir: Path, ui=None):
    debug_dir.mkdir(parents=True, exist_ok=True)
    img = frame.copy()
    if ui is not None:
        for region, color in [
            (ui.timer,          (0, 255, 0)),
            (ui.elixir_number,  (255, 0, 0)),
            (ui.multiplier_icon,(0, 165, 255)),
        ]:
            x1, y1, x2, y2 = region.to_tuple()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite(str(debug_dir / f"{label}.jpg"), img)


def process_file(
    path: Path,
    preload_ocr: bool = False,
    dry_run: bool = False,
    override_video: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
) -> bool:
    print(f"Processing {path.name}")
    try:
        text = path.read_text()
        if not text or not text.strip():
            print(f"  Skipping empty file")
            return False
        data = json.loads(text)
    except Exception as e:
        print(f"  Failed to read JSON: {e}")
        return False

    if "video_info" not in data or "frames" not in data:
        print(f"  Skipping: missing video_info or frames")
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
                if p.stem.lower() == stem:
                    found = p
                    break
                if stem in p.stem.lower():
                    candidates.append(p)
            if found is None and candidates:
                candidates.sort(key=lambda x: len(x.name))
                found = candidates[0]
        if found:
            video_path = found
        else:
            print(f"  Video not found: {video_path}")
            return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Failed to open video: {video_path}")
        return False

    vid_w = int(data["video_info"].get("width", 0))
    vid_h = int(data["video_info"].get("height", 0))
    if vid_w == 0 or vid_h == 0:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False
        vid_h, vid_w = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detector = DigitDetector(preload_ocr=preload_ocr)
    timestamps = [int(f.get("timestamp_ms", 0)) for f in data["frames"]]
    n = len(timestamps)

    timer_detected:      List[Optional[int]] = [None] * n
    elixir_detected:     List[Optional[int]] = [None] * n
    multiplier_detected: List[Optional[int]] = [None] * n

    prev_frame = None
    prev_ui = None
    prev_elixir: Optional[int] = None
    file_debug = (debug_dir / path.stem) if debug_dir else None

    for i, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        ui = UIRegions(vid_w, vid_h)

        # Timer
        ui.align_timer_to_image(frame)
        secs = detector.detect_timer(frame, ui.timer.to_tuple())
        if secs is not None:
            timer_detected[i] = int(secs)

        # Elixir
        ui.align_elixir_to_image(frame)
        elixir_result = detector.detect_elixir(frame, ui.elixir_number.to_tuple())
        if elixir_result.detected:
            elixir_detected[i] = int(elixir_result.value)

        # Multiplier
        ui.align_multiplier_to_image(frame)
        mult = detector.detect_multiplier_icon(frame, ui.multiplier_icon.to_tuple())
        if mult > 1:
            multiplier_detected[i] = mult

        t_str = str(secs) if secs is not None else "?"
        e_str = str(elixir_detected[i]) if elixir_detected[i] is not None else "?"
        print(f"  [{i}] ts={ts}ms  timer={t_str}  elixir={e_str}  x{mult}")

        if file_debug:
            if (prev_elixir is not None and elixir_detected[i] is not None
                    and prev_elixir - elixir_detected[i] >= 5):
                _save_debug(prev_frame, f"elixir_drop_{i-1}_was{prev_elixir}", file_debug, prev_ui)
                _save_debug(frame,      f"elixir_drop_{i}_now{elixir_detected[i]}", file_debug, ui)

        if elixir_detected[i] is not None:
            prev_elixir = elixir_detected[i]
        prev_frame = frame.copy()
        prev_ui = ui

    cap.release()

    n_timer      = sum(1 for v in timer_detected if v is not None)
    n_elixir     = sum(1 for v in elixir_detected if v is not None)
    n_multiplier = sum(1 for v in multiplier_detected if v is not None)
    print(f"  Detected timer: {n_timer}/{n}, elixir: {n_elixir}/{n}, multiplier>1: {n_multiplier}/{n}")

    # Discard all frames before the first timer detection (loading screen)
    first_timer = next((i for i, v in enumerate(timer_detected) if v is not None), None)
    if first_timer is None:
        print(f"  No timer detections — skipping file entirely")
        return False
    if first_timer > 0:
        print(f"  Trimming {first_timer} pre-game frames (loading screen)")
        data["frames"]       = data["frames"][first_timer:]
        timer_detected       = timer_detected[first_timer:]
        elixir_detected      = elixir_detected[first_timer:]
        multiplier_detected  = multiplier_detected[first_timer:]
        timestamps           = timestamps[first_timer:]
        n                    = len(timestamps)

    timer_filled      = interpolate_timer(timestamps, timer_detected)
    elixir_filled     = fill_nearest(elixir_detected)
    multipliers, phases = derive_phase_fields(timestamps, timer_filled, multiplier_detected)

    changed = False
    for i in range(n):
        fr = data["frames"][i]
        if timer_filled[i] is not None and fr.get("game_time_remaining") != timer_filled[i]:
            fr["game_time_remaining"] = timer_filled[i]
            changed = True
        if elixir_filled[i] is not None and fr.get("player_elixir") != elixir_filled[i]:
            fr["player_elixir"] = elixir_filled[i]
            changed = True
        if fr.get("elixir_multiplier") != multipliers[i]:
            fr["elixir_multiplier"] = multipliers[i]
            changed = True
        if fr.get("game_phase") != phases[i]:
            fr["game_phase"] = phases[i]
            changed = True

    if changed:
        if dry_run:
            dry_dir = path.parent / "dryrun"
            dry_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(dry_dir / path.name, data)
            print(f"  Dry-run: wrote {dry_dir / path.name}")
        else:
            atomic_write_json(path, data)
            print(f"  Updated {path.name}")
    else:
        print(f"  No changes")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses-dir", default="output/analysis")
    parser.add_argument("--videos-dir", default="data/replays")
    parser.add_argument("--pairing", choices=["name", "order"], default="name")
    parser.add_argument("--preload-ocr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-dir", default=None,
                        help="Save debug images for elixir drops and unknown sequences here")
    args = parser.parse_args()

    analyses_base = Path(args.analyses_dir)
    videos_base   = Path(args.videos_dir)
    if not analyses_base.exists():
        print(f"Analyses dir not found: {analyses_base}")
        raise SystemExit(2)
    if not videos_base.exists():
        print(f"Videos dir not found: {videos_base}")
        raise SystemExit(2)

    analyses = sorted(p for p in analyses_base.rglob("*_analysis.json") if p.is_file())
    videos   = sorted(p for p in videos_base.rglob("*") if p.is_file())

    if not analyses:
        print(f"No *_analysis.json files found under {analyses_base}")
        return 0

    pairs = []
    if args.pairing == "order":
        for i in range(min(len(analyses), len(videos))):
            pairs.append((analyses[i], videos[i]))
    else:
        video_map = {v.stem.lower(): v for v in videos}
        for a in analyses:
            base = a.stem[:-9] if a.stem.endswith("_analysis") else a.stem
            found = video_map.get(base.lower())
            if not found:
                for vstem, vp in video_map.items():
                    if base.lower() in vstem:
                        found = vp
                        break
            pairs.append((a, found))

    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    print(f"Processing {len(pairs)} files")
    for analysis_path, video_path in pairs:
        process_file(
            analysis_path,
            preload_ocr=args.preload_ocr,
            dry_run=args.dry_run,
            override_video=video_path,
            debug_dir=debug_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

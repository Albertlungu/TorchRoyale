"""
Full pipeline test for model + OCR integration.

Covers:
  - On-field models:
      - Cicadas (player cards, hog-cycle classes)
      - Visionbot (opponent cards, enemies classes)
  - Hand classifier
  - Tower HP OCR + win/loss inference
  - Player elixir OCR
  - Timer OCR
  - Multiplier icon OCR + derived game phase tracking

Runs detection every --stride frames starting at --start on a specified video.
Saves annotated frames to output/test_frames/.

Usage:
  python tests/test_full_pipeline.py data/replays/Game_23.mp4
  python tests/test_full_pipeline.py data/replays/Game_23.mp4 --start 1000 --stride 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.detection.dual_model_detector import DualModelDetector
from src.detection.hand_classifier import HandClassifier, get_next_bbox
from src.detection.katacr_legacy import KataCRDetector
from src.detection.result import Detection
from src.game_state.phase import derive_phases
from src.ocr.detector import DigitDetector
from src.ocr.regions import UIRegions
from src.ocr.tower_tracker import TowerTracker, TowerReading, determine_outcome_from_readings

_OUTPUT_DIR = Path("output/test_frames")
_DEFAULT_CICADAS = Path("data/models/onfield/hog-cycle-detector-best.pt")
_DEFAULT_VISIONBOT = Path("data/models/onfield/torchroyale-enemies-best.pt")
_DEFAULT_HAND = Path("data/models/hand_classifier/hand_classifier.pt")

# Bbox colours (BGR)
_CICADAS_COLOUR  = (0, 200, 0)    # green:  player cards
_VISIONBOT_COLOUR = (0, 0, 220)   # red:    opponent cards
_HAND_COLOUR     = (200, 140, 0)  # orange: hand slots
_TOWER_OPP_COLOUR = (0, 80, 220)  # red-ish: opponent towers
_TOWER_PLR_COLOUR = (0, 220, 80)  # green-ish: player towers
_TIMER_COLOUR = (220, 220, 0)     # cyan-ish
_ELIXIR_COLOUR = (200, 0, 200)    # magenta
_MULTI_COLOUR = (255, 140, 0)     # orange/blue

_TOWER_DEFS = [
    ("opp_left",  "opponent_tower_left",  _TOWER_OPP_COLOUR, False),
    ("opp_king",  "opponent_tower_king",  _TOWER_OPP_COLOUR, True),
    ("opp_right", "opponent_tower_right", _TOWER_OPP_COLOUR, False),
    ("pl_left",   "player_tower_left",    _TOWER_PLR_COLOUR, False),
    ("pl_king",   "player_tower_king",    _TOWER_PLR_COLOUR, True),
    ("pl_right",  "player_tower_right",   _TOWER_PLR_COLOUR, False),
]


def _draw_detections(
    frame: np.ndarray,
    on_field: List[Detection],
    hand_labels: List[Optional[str]],
    game_strip: Optional[Tuple[int, int]],
    ui: UIRegions,
    tower_readings: Dict[str, TowerReading],
    timer_secs: Optional[int],
    player_elixir: Optional[int],
    multiplier_icon: int,
    phase_mult: int,
    phase_name: str,
) -> np.ndarray:
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    # On-field detections
    for det in on_field:
        if not det.bbox_px:
            continue
        x1, y1, x2, y2 = det.bbox_px
        colour = _VISIONBOT_COLOUR if det.is_opponent else _CICADAS_COLOUR
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    # Hand slots
    if game_strip is not None:
        x_left, x_right = game_strip
        game_w = x_right - x_left
        from src.detection.hand_classifier import (
            _INSET_FRAC, _NEXT_RIGHT_FRAC, _SLOT_OFFSET_FRACS,
            _Y_BOT_FRAC, _Y_TOP_FRAC,
        )
        next_end = x_left + int(game_w * _NEXT_RIGHT_FRAC)
        y_top = int(frame_h * _Y_TOP_FRAC)
        y_bot = int(frame_h * _Y_BOT_FRAC)
        cards_w = x_right - next_end
        slot_w = cards_w // 4
        inset = int(slot_w * _INSET_FRAC)
        for i, (lbl, offset_frac) in enumerate(zip(hand_labels, _SLOT_OFFSET_FRACS)):
            offset = int(slot_w * offset_frac)
            sx1 = next_end + i * slot_w + inset + offset
            sx2 = sx1 + slot_w - 2 * inset
            sx1, sx2 = max(0, sx1), min(frame_w, sx2)
            cv2.rectangle(out, (sx1, y_top), (sx2, y_bot), _HAND_COLOUR, 2)
            if lbl:
                cv2.putText(out, lbl, (sx1, y_top - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _HAND_COLOUR, 1, cv2.LINE_AA)
        nx1, ny1, nx2, ny2 = get_next_bbox(frame_h, frame_w, x_left, x_right)
        cv2.rectangle(out, (nx1, ny1), (nx2, ny2), _HAND_COLOUR, 1)

    # Tower regions
    for label, attr, colour, _ in _TOWER_DEFS:
        region = getattr(ui, attr)
        x1, y1, x2, y2 = region.to_tuple()
        reading = tower_readings.get(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        if reading:
            if reading.destroyed:
                txt = "DESTROYED"
            elif reading.at_max:
                txt = f"MAX({reading.hp})"
            else:
                txt = str(reading.hp) if reading.hp is not None else "?"
            cv2.putText(out, txt, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    # Timer / elixir / multiplier OCR regions
    tx1, ty1, tx2, ty2 = ui.timer.to_tuple()
    ex1, ey1, ex2, ey2 = ui.elixir_number.to_tuple()
    mx1, my1, mx2, my2 = ui.multiplier_icon.to_tuple()
    cv2.rectangle(out, (tx1, ty1), (tx2, ty2), _TIMER_COLOUR, 2)
    cv2.rectangle(out, (ex1, ey1), (ex2, ey2), _ELIXIR_COLOUR, 2)
    cv2.rectangle(out, (mx1, my1), (mx2, my2), _MULTI_COLOUR, 2)
    timer_txt = "?:??" if timer_secs is None else f"{timer_secs // 60}:{timer_secs % 60:02d}"
    elixir_txt = "?" if player_elixir is None else str(player_elixir)
    header = (
        f"timer={timer_txt}  elixir={elixir_txt}  "
        f"mult(icon/phase)={multiplier_icon}/{phase_mult}  phase={phase_name}"
    )
    cv2.putText(
        out, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        out, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA
    )

    return out


def _detect_game_strip(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
    cols = np.where(np.mean(gray, axis=0) > 30)[0]
    if cols.size == 0:
        return None
    return int(cols.min()), int(cols.max())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full pipeline smoke test.")
    parser.add_argument("video", help="Path to input video file.")
    parser.add_argument(
        "--onfield-backend",
        choices=["dual", "katacr-legacy"],
        default="dual",
        help="On-field detector backend. 'dual' uses cicadas+opponent model; 'katacr-legacy' uses KataCR detector1+detector2.",
    )
    parser.add_argument("--cicadas-weights", default=str(_DEFAULT_CICADAS))
    parser.add_argument("--visionbot-weights", default=str(_DEFAULT_VISIONBOT))
    parser.add_argument("--hand-weights", default=str(_DEFAULT_HAND))
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--count",  type=int, default=20)
    parser.add_argument("--conf",   type=float, default=0.1,
                        help="Confidence threshold for on-field detections.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)
    cicadas_weights = Path(args.cicadas_weights)
    visionbot_weights = Path(args.visionbot_weights)
    hand_weights = Path(args.hand_weights)
    required = [("hand classifier", hand_weights)]
    if args.onfield_backend == "dual":
        required.extend([
            ("cicadas", cicadas_weights),
            ("visionbot", visionbot_weights),
        ])
    for label, path in required:
        if not path.exists():
            print(f"Error: {label} weights not found: {path}")
            sys.exit(1)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing detectors...")
    if args.onfield_backend == "katacr-legacy":
        detector = KataCRDetector()
    else:
        detector = DualModelDetector(
            cicadas_weights=str(cicadas_weights),
            visionbot_weights=str(visionbot_weights),
            conf_threshold=args.conf,
        )
    hand_clf = HandClassifier(weights_path=str(hand_weights))
    ocr = DigitDetector()

    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vid_w}x{vid_h} @ {fps}fps, {total} frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, cal_frame = cap.read()
    if not ok:
        print(f"Error: could not read frame {args.start}")
        sys.exit(1)

    print(f"Calibrating from frame {args.start}...")
    detector.calibrate(cal_frame)

    ui = UIRegions(vid_w, vid_h)
    trackers = {
        label: TowerTracker(is_king=is_king)
        for label, _, _, is_king in _TOWER_DEFS
    }
    last_hp  = {label: None for label, *_ in _TOWER_DEFS}
    timer_filled: List[Optional[int]] = []
    mult_mismatch = 0

    saved = 0
    frame_idx = args.start

    while saved < args.count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        ui.align_timer(frame)
        ui.align_elixir(frame)
        ui.align_multiplier(frame)
        game_strip = _detect_game_strip(frame)
        timer_secs = ocr.detect_timer(frame, ui.timer.to_tuple())
        elixir_res = ocr.detect_elixir(frame, ui.elixir_number.to_tuple())
        icon_mult = ocr.detect_multiplier(frame, ui.multiplier_icon.to_tuple())
        player_elixir = elixir_res.value if elixir_res.detected else None
        timer_filled.append(timer_secs if timer_secs is not None else (timer_filled[-1] if timer_filled else None))
        derived_mults, phases = derive_phases(timer_filled)
        phase_mult = derived_mults[-1] if derived_mults else 1
        phase_name = phases[-1] if phases else "single"
        if icon_mult != phase_mult:
            mult_mismatch += 1

        field_result = detector.detect(frame)
        hand_labels  = hand_clf.classify(frame, game_strip=game_strip)

        # Tower OCR — pass 1: princess towers to establish levels
        for label, attr, _, is_king in _TOWER_DEFS:
            if not is_king:
                raw = ocr.detect_tower_raw(frame, getattr(ui, attr).to_tuple())
                trackers[label].read(raw)

        # Propagate princess level to king before reading kings
        for king_lbl, left_lbl, right_lbl in [
            ("pl_king", "pl_left", "pl_right"),
            ("opp_king", "opp_left", "opp_right"),
        ]:
            for src in [left_lbl, right_lbl]:
                if trackers[src].level is not None:
                    trackers[king_lbl].set_level(trackers[src].level)
                    break

        # Pass 2: read all towers
        tower_readings: Dict[str, TowerReading] = {}
        for label, attr, _, is_king in _TOWER_DEFS:
            raw = ocr.detect_tower_raw(
                frame, getattr(ui, attr).to_tuple(),
                bottom_fraction=0.7 if is_king else 1.0,
                invert=is_king,
            )
            reading = trackers[label].read(raw)
            tower_readings[label] = reading
            if reading.hp is not None:
                last_hp[label] = reading.hp

        annotated = _draw_detections(
            frame,
            field_result.on_field,
            hand_labels,
            game_strip,
            ui,
            tower_readings,
            timer_secs,
            player_elixir,
            icon_mult,
            phase_mult,
            phase_name,
        )

        out_path = _OUTPUT_DIR / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(out_path), annotated)

        player_dets = sum(1 for d in field_result.on_field if not d.is_opponent)
        opp_dets    = sum(1 for d in field_result.on_field if d.is_opponent)
        hand_str    = ", ".join(l or "?" for l in hand_labels)
        timer_str = "?:??" if timer_secs is None else f"{timer_secs // 60}:{timer_secs % 60:02d}"
        elixir_str = "?" if player_elixir is None else str(player_elixir)
        pl_hp_str   = " | ".join(
            f"{l.split('_')[1]}={tower_readings[l].hp or '?'}"
            for l in ["pl_left", "pl_king", "pl_right"]
        )
        opp_hp_str  = " | ".join(
            f"{l.split('_')[1]}={tower_readings[l].hp or '?'}"
            for l in ["opp_left", "opp_king", "opp_right"]
        )
        print(
            f"frame {frame_idx:5d}  "
            f"({player_dets}pl/{opp_dets}opp on-field)  "
            f"hand=[{hand_str}]  "
            f"timer={timer_str}  "
            f"elixir={elixir_str}  "
            f"mult(icon/phase)={icon_mult}/{phase_mult} phase={phase_name}  "
            f"pl_towers=[{pl_hp_str}]  opp_towers=[{opp_hp_str}]"
        )

        saved += 1
        frame_idx += args.stride

    cap.release()

    # Final outcome
    p_labels = ["pl_left", "pl_king", "pl_right"]
    o_labels = ["opp_left", "opp_king", "opp_right"]
    outcome = determine_outcome_from_readings(
        [last_hp[l] for l in p_labels],
        [trackers[l].destroyed for l in p_labels],
        [last_hp[l] for l in o_labels],
        [trackers[l].destroyed for l in o_labels],
    )
    print(f"\nDone! {saved} frames saved to {_OUTPUT_DIR}")
    print(f"Inferred outcome: {outcome.upper()}")
    print(f"Multiplier disagreements (icon vs timer-derived): {mult_mismatch}/{saved}")


if __name__ == "__main__":
    main()

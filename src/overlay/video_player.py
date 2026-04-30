"""
Video player with tile-highlight overlay for replay analysis.

Uses OpenCV for display (reliable on macOS). Reads a JSONL recommendation
log and draws a glowing green rectangle on the recommended tile as the
video plays.

Controls:
  Space / k  -- play / pause
  h          -- toggle overlay on/off
  q / Escape -- quit
"""

import bisect
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.grid.coordinate_mapper import CoordinateMapper


# Maximum display height (video is scaled to fit).
_DISPLAY_HEIGHT = 800

# Overlay colors (BGR for OpenCV).
_COLOR_READY = (68, 255, 0)       # bright green
_COLOR_WAIT = (128, 128, 128)     # gray
_COLOR_FILL_READY = (17, 51, 0)   # dark green
_COLOR_FILL_WAIT = (40, 40, 40)   # dark gray
_COLOR_TEXT_BG = (0, 0, 0)        # black
_COLOR_STATUS_BG = (34, 34, 34)   # dark gray
_COLOR_STATUS_FG = (255, 255, 255)


class VideoPlayer:
    """
    Plays a replay video with an overlay showing DT recommendations.

    Args:
        video_path:  Path to the replay MP4.
        jsonl_path:  Path to the JSONL recommendation log.
    wait_state:  "debug"      -- always show overlay regardless of elixir.
                 "production" -- dim overlay when elixir is insufficient.

    Attributes:
        _cap (cv2.VideoCapture): OpenCV video capture.
        _fps (float): Video frames per second.
        _total_frames (int): Total frames in video.
        _vid_w (int): Video width.
        _vid_h (int): Video height.
        _scale (float): Display scaling factor.
        _disp_w (int): Display width.
        _disp_h (int): Display height.
        _frame_delay_ms (int): Delay between frames in milliseconds.
        _mapper (CoordinateMapper): Coordinate mapper.
        _timestamps (List[int]): Timestamps from JSONL.
        _all_entries (List[Dict]): All JSONL entries.
        _rec_entries (List[Dict]): Entries with recommendations.
        _rec_timestamps (List[int]): Timestamps with recommendations.
        _playing (bool): Whether video is playing.
        _overlay_visible (bool): Whether overlay is shown.
        _last_printed_detections (Optional[str]): Last printed detection summary.
        _wait_state (str): "debug" or "production".
    """

    def __init__(
        self,
        video_path: str,
        jsonl_path: str,
        wait_state: str = "debug",
    ):
        if wait_state not in ("debug", "production"):
            raise ValueError(f"wait_state must be 'debug' or 'production', got {wait_state!r}")

        self._wait_state = wait_state

        print(f"[VideoPlayer] Opening video: {video_path}", flush=True)
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._vid_w == 0 or self._vid_h == 0:
            raise RuntimeError(f"Video reports zero dimensions: {self._vid_w}x{self._vid_h}")

        self._scale = _DISPLAY_HEIGHT / self._vid_h
        self._disp_w = int(self._vid_w * self._scale)
        self._disp_h = _DISPLAY_HEIGHT
        self._frame_delay_ms = max(1, int(1000 / self._fps))

        print(f"[VideoPlayer] Video: {self._vid_w}x{self._vid_h} @ {self._fps:.1f} fps, "
              f"{self._total_frames} frames", flush=True)
        print(f"[VideoPlayer] Display: {self._disp_w}x{self._disp_h}, "
              f"frame_delay={self._frame_delay_ms}ms", flush=True)

        # Coordinate mapper calibrated to VIDEO resolution (not display).
        # Sample a frame from 10% into the video to avoid loading screens,
        # then use it to detect game content bounds for portrait-in-landscape.
        self._mapper = CoordinateMapper()
        _calib_pos = max(0, int(self._total_frames * 0.10))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, _calib_pos)
        ret, _calib_frame = self._cap.read()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ret and _calib_frame is not None:
            self._mapper.calibrate_from_frame(_calib_frame)
        else:
            self._mapper.calibrate_from_image(self._vid_w, self._vid_h)

        # Load JSONL.
        all_entries = _load_jsonl(jsonl_path)
        self._timestamps: List[int] = [e["timestamp_ms"] for e in all_entries]
        self._all_entries: List[Dict] = all_entries
        self._rec_entries: List[Dict] = [e for e in all_entries if e.get("has_recommendation")]
        self._rec_timestamps: List[int] = [e["timestamp_ms"] for e in self._rec_entries]

        n_recs = len(self._rec_entries)
        print(f"[VideoPlayer] Loaded {len(all_entries)} JSONL entries "
              f"({n_recs} with recommendations)", flush=True)

        # Playback state.
        self._playing = True
        self._overlay_visible = True
        self._last_printed_detections: Optional[str] = None  # dedup detection prints

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _entry_at(self, timestamp_ms: int) -> Optional[Dict]:
        """Return the nearest JSONL entry at or before timestamp_ms."""
        idx = bisect.bisect_right(self._timestamps, timestamp_ms) - 1
        if idx < 0:
            return None
        return self._all_entries[idx]

    def _elixir_at(self, timestamp_ms: int) -> int:
        entry = self._entry_at(timestamp_ms)
        if entry is None:
            return 0
        return entry.get("player_elixir", 0)

    def _rec_at(self, timestamp_ms: int) -> Optional[Dict]:
        idx = bisect.bisect_right(self._rec_timestamps, timestamp_ms) - 1
        if idx < 0:
            return None
        return self._rec_entries[idx]

    # ------------------------------------------------------------------
    # Overlay drawing (draws directly onto the frame in video coords)
    # ------------------------------------------------------------------

    def _draw_overlay(
        self,
        disp_frame: np.ndarray,
        rec: Optional[Dict],
        current_elixir: int,
    ):
        """
        Draw the tile highlight and card label onto the display-resolution frame.

        disp_frame is already resized to (disp_w, disp_h). All coordinates are
        scaled from video coords to display coords before drawing so that text
        and borders render at a consistent size regardless of source resolution.
        """
        if not self._overlay_visible or rec is None:
            return

        tile_x: int = rec["tile_x"]
        tile_y: int = rec["tile_y"]
        card: str = rec["card"]
        elixir_required: int = rec.get("elixir_required", 0)

        ready = True
        if self._wait_state == "production":
            ready = current_elixir >= elixir_required

        # Tile bounding box in VIDEO coords, then scale to display coords.
        vx0, vy0, vx1, vy1 = self._mapper.get_tile_bounds_pixels(tile_x, tile_y)
        s = self._scale
        x0, y0, x1, y1 = int(vx0 * s), int(vy0 * s), int(vx1 * s), int(vy1 * s)

        color = _COLOR_READY if ready else _COLOR_WAIT
        fill = _COLOR_FILL_READY if ready else _COLOR_FILL_WAIT

        # Semi-transparent fill.
        overlay = disp_frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1)
        cv2.addWeighted(overlay, 0.4, disp_frame, 0.6, 0, dst=disp_frame)

        # Outer glow (3 concentric rectangles).
        for pad in range(3, 0, -1):
            alpha = 0.3 + 0.2 * (3 - pad)
            glow = disp_frame.copy()
            cv2.rectangle(glow, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), color, 1)
            cv2.addWeighted(glow, alpha, disp_frame, 1.0 - alpha, 0, dst=disp_frame)

        # Main border.
        cv2.rectangle(disp_frame, (x0, y0), (x1, y1), color, 2)

        # Strip Roboflow suffixes for display.
        display_card = _strip_card_suffix(card)
        label = display_card if ready else f"{display_card} (wait)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        h_disp, w_disp = disp_frame.shape[:2]
        pad = 4

        # Center label over the tile, clamped to stay within the frame.
        label_x = x0 + (x1 - x0 - tw) // 2
        label_x = max(pad, min(label_x, w_disp - tw - pad * 2))

        # Place above the tile if there's room, otherwise below.
        if y0 > th + pad * 2 + 6:
            label_y = y0 - 6
        elif y1 + th + pad * 2 + 6 < h_disp:
            label_y = y1 + th + 6
        else:
            label_y = y0 + th  # fallback: inside tile

        cv2.rectangle(
            disp_frame,
            (label_x - pad, label_y - th - pad),
            (label_x + tw + pad, label_y + baseline + pad),
            _COLOR_TEXT_BG,
            -1,
        )
        cv2.putText(disp_frame, label, (label_x, label_y),
                    font, font_scale, color, thickness)

    def _draw_status_bar(self, frame: np.ndarray, text: str):
        """Draw a status bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        bar_h = 30
        cv2.rectangle(frame, (0, h - bar_h), (w, h), _COLOR_STATUS_BG, -1)
        cv2.putText(
            frame, text,
            (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_STATUS_FG, 1,
        )

    def _print_detections(self, entry: Optional[Dict], timestamp_ms: int):
        """Print field detections to the terminal when they change."""
        if entry is None:
            return

        detections = entry.get("detections", [])
        if not detections:
            return

        # Build a short summary string and only print when it changes.
        on_field = [d for d in detections if d.get("is_on_field", False)]
        in_hand = [d for d in detections if not d.get("is_on_field", False)
                   and not d.get("is_opponent", False)]

        parts = []
        if in_hand:
            hand_names = [_strip_card_suffix(d["class_name"]) for d in in_hand]
            parts.append(f"Hand: {', '.join(hand_names)}")
        if on_field:
            player_field = [d for d in on_field if not d.get("is_opponent", False)]
            opp_field = [d for d in on_field if d.get("is_opponent", False)]
            if player_field:
                names = [f"{_strip_card_suffix(d['class_name'])}@({d['tile_x']},{d['tile_y']})"
                         for d in player_field]
                parts.append(f"Player field: {', '.join(names)}")
            if opp_field:
                names = [f"{_strip_card_suffix(d['class_name'])}@({d['tile_x']},{d['tile_y']})"
                         for d in opp_field]
                parts.append(f"Opp field: {', '.join(names)}")

        summary = " | ".join(parts)
        if summary != self._last_printed_detections:
            sec = timestamp_ms / 1000
            print(f"[{sec:6.1f}s] {summary}", flush=True)
            self._last_printed_detections = summary

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def run(self):
        """Run the player loop."""
        window_name = "TorchRoyale Replay"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self._disp_w, self._disp_h)

        print("[VideoPlayer] Playback started. Press Q to quit.", flush=True)
        frame_count = 0

        while True:
            if self._playing:
                timestamp_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
                ret, frame = self._cap.read()

                if not ret:
                    # Show "end of video" on a black frame and wait for quit.
                    end_frame = np.zeros((self._vid_h, self._vid_w, 3), dtype=np.uint8)
                    self._draw_status_bar(end_frame, "End of video  |  Q: quit")
                    resized = cv2.resize(end_frame, (self._disp_w, self._disp_h))
                    cv2.imshow(window_name, resized)
                    print("[VideoPlayer] End of video.", flush=True)
                    # Wait for quit key.
                    while True:
                        key = cv2.waitKey(100) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    break

                frame_count += 1
                if frame_count == 1:
                    print(f"[VideoPlayer] First frame: shape={frame.shape}", flush=True)

                # Look up current state and recommendation.
                entry = self._entry_at(timestamp_ms)
                current_elixir = entry.get("player_elixir", 0) if entry else 0
                rec = self._rec_at(timestamp_ms)

                # Print detections when they change.
                self._print_detections(entry, timestamp_ms)

                # Resize first, then draw overlay at display resolution so
                # text and borders render at a consistent, legible size.
                resized = cv2.resize(frame, (self._disp_w, self._disp_h))
                self._draw_overlay(resized, rec, current_elixir)

                # Status bar.
                sec = timestamp_ms / 1000
                card_label = _strip_card_suffix(rec["card"]) if rec else "-"
                overlay_tag = "ON" if self._overlay_visible else "OFF"
                status = (f"t={sec:.1f}s  |  elixir={current_elixir}  |  "
                          f"rec={card_label}  |  overlay={overlay_tag}  |  "
                          f"Space:pause  H:overlay  Q:quit")
                self._draw_status_bar(resized, status)

                cv2.imshow(window_name, resized)

            # Handle keyboard input.
            key = cv2.waitKey(self._frame_delay_ms if self._playing else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord(" "), ord("k")):
                self._playing = not self._playing
                state = "playing" if self._playing else "paused"
                print(f"[VideoPlayer] {state}", flush=True)
            elif key == ord("h"):
                self._overlay_visible = not self._overlay_visible
                state = "ON" if self._overlay_visible else "OFF"
                print(f"[VideoPlayer] Overlay: {state}", flush=True)

        self._cap.release()
        cv2.destroyAllWindows()
        print("[VideoPlayer] Done.", flush=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_CARD_SUFFIXES = (
    "-in-hand", "-next", "-on-field", "_on_field",
    "-evolution", "_evolution", "-ability",
)

def _strip_card_suffix(name: str) -> str:
    """Remove Roboflow detection suffixes for cleaner display."""
    low = name.lower()
    for suffix in _CARD_SUFFIXES:
        if low.endswith(suffix):
            return name[: len(name) - len(suffix)]
    return name


def _load_jsonl(path: str) -> List[Dict]:
    entries: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: e["timestamp_ms"])
    return entries

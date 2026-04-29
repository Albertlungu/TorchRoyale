"""
VideoPlayer: replay viewer with DT recommendation overlay.
Reads the JSONL produced by InferenceRunner and draws card placements.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.grid.coordinate_mapper import CoordinateMapper

_DISPLAY_HEIGHT = 800
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class VideoPlayer:
    def __init__(self, video_path: str, jsonl_path: str, production: bool = False):
        self._video_path = video_path
        self._production = production

        self._cap = cv2.VideoCapture(video_path)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._scale = _DISPLAY_HEIGHT / self._vid_h
        self._disp_w = int(self._vid_w * self._scale)
        self._disp_h = _DISPLAY_HEIGHT
        self._frame_delay = max(1, int(1000 / self._fps))

        # Coordinate mapper — calibrate from a mid-game frame
        self._mapper = CoordinateMapper()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(self._total * 0.10)))
        ret, calib = self._cap.read()
        if ret:
            self._mapper.calibrate_from_frame(calib)
        else:
            self._mapper.calibrate_from_image(self._vid_w, self._vid_h)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        entries = _load_jsonl(jsonl_path)
        self._timestamps = [e["timestamp_ms"] for e in entries]
        self._entries = entries
        self._rec_entries = [e for e in entries if e.get("has_recommendation")]

        print(f"[VideoPlayer] {self._vid_w}x{self._vid_h} @ {self._fps:.1f} fps")
        print(f"[VideoPlayer] {len(entries)} frames, {len(self._rec_entries)} with recommendations")

    def _draw_overlay(self, frame: np.ndarray, entry: Dict) -> np.ndarray:
        if not entry.get("has_recommendation"):
            return frame
        card  = entry.get("card", "")
        tx    = entry.get("tile_x", 0)
        ty    = entry.get("tile_y", 0)
        elixir_req = entry.get("elixir_required", 0)

        if self._mapper.bounds is None:
            return frame

        x1, y1, x2, y2 = self._mapper.tile_bounds_pixels(tx, ty)
        s = self._scale
        dx1, dy1 = int(x1 * s), int(y1 * s)
        dx2, dy2 = int(x2 * s), int(y2 * s)

        player_elixir = entry.get("player_elixir", 0) or 0
        dim = self._production and player_elixir < elixir_req
        color = (0, 140, 255) if not dim else (60, 60, 60)

        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
        label = card
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.4, 1)
        cv2.rectangle(frame, (dx1, dy1 - th - 4), (dx1 + tw + 4, dy1), color, -1)
        cv2.putText(frame, label, (dx1 + 2, dy1 - 2), _FONT, 0.4, (255, 255, 255), 1)
        return frame

    def _find_entry(self, ts_ms: int) -> Optional[Dict]:
        """Find the most recent entry at or before ts_ms."""
        lo, hi = 0, len(self._timestamps) - 1
        result = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._timestamps[mid] <= ts_ms:
                result = self._entries[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def _status_bar(self, frame: np.ndarray, entry: Optional[Dict]) -> np.ndarray:
        h, w = frame.shape[:2]
        bar = np.zeros((24, w, 3), dtype=np.uint8)
        if entry:
            ts = entry["timestamp_ms"] / 1000
            elx = entry.get("player_elixir", 0)
            card = entry.get("card", "") if entry.get("has_recommendation") else "—"
            text = f"t={ts:.1f}s  elixir={elx}  rec={card}  overlay=ON  Space:pause H:overlay Q:quit"
        else:
            text = "Space:pause  H:overlay  Q:quit"
        cv2.putText(bar, text, (4, 16), _FONT, 0.38, (200, 200, 200), 1)
        return np.vstack([frame, bar])

    def run(self) -> None:
        window = "TorchRoyale Replay"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, self._disp_w, self._disp_h + 24)

        show_overlay = True
        paused = False
        current_entry: Optional[Dict] = None

        while True:
            if not paused:
                ret, frame = self._cap.read()
                if not ret:
                    break
                ts_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
                current_entry = self._find_entry(ts_ms)

            disp = cv2.resize(frame, (self._disp_w, self._disp_h))
            if show_overlay and current_entry:
                disp = self._draw_overlay(disp, current_entry)

            # Hand / field overlay text
            if current_entry:
                hand = current_entry.get("detections", [])
                in_hand = [d["class_name"].replace("-in-hand","") for d in hand if "-in-hand" in d.get("class_name","")]
                on_field_p = [f"{d['class_name']}@({d['tile_x']},{d['tile_y']})" for d in hand if d.get("is_on_field") and not d.get("is_opponent")]
                on_field_o = [f"{d['class_name']}@({d['tile_x']},{d['tile_y']})" for d in hand if d.get("is_on_field") and d.get("is_opponent")]
                ts = current_entry["timestamp_ms"] / 1000
                if in_hand:
                    print(f"[{ts:6.1f}s] Hand: {', '.join(in_hand)} | Player field: {', '.join(on_field_p)}" + (f" | Opp field: {', '.join(on_field_o)}" if on_field_o else ""))

            disp = self._status_bar(disp, current_entry)
            cv2.imshow(window, disp)

            key = cv2.waitKey(self._frame_delay if not paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" ") or key == ord("k"):
                paused = not paused
            elif key == ord("h"):
                show_overlay = not show_overlay

        self._cap.release()
        cv2.destroyAllWindows()
        print("[VideoPlayer] Done.")

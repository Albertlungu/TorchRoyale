"""
VideoPlayer: replay viewer with Decision Transformer recommendation overlay.

Reads the JSONL file produced by InferenceRunner, synchronises entries to
the video timestamp, and draws card placement suggestions on the frame.

Controls:
  Space / k  -- pause / resume
  h          -- toggle overlay visibility
  q / Esc    -- quit

Public API:
  VideoPlayer -- construct with video + JSONL paths, call run() to open the window
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.grid.coordinate_mapper import CoordinateMapper
from src.types import RecommendationDict

_DISPLAY_HEIGHT: int = 800
_FONT: int = cv2.FONT_HERSHEY_SIMPLEX


def _load_jsonl(path: str) -> List[RecommendationDict]:
    """
    Load a JSONL file into a list of RecommendationDicts.

    Args:
        path: path to the .jsonl file.

    Returns:
        List of entries, one per line.
    """
    entries: List[RecommendationDict] = []
    with open(path, encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class VideoPlayer:
    """
    Replay viewer that overlays DT card placement recommendations on the video.

    In production mode, the overlay dims the recommendation box when the player
    does not yet have enough elixir to afford the suggested card.
    """

    def __init__(
        self, video_path: str, jsonl_path: str, production: bool = False
    ) -> None:
        """
        Args:
            video_path:  path to the source replay video.
            jsonl_path:  path to the InferenceRunner JSONL output.
            production:  if True, dim recommendations when elixir is insufficient.
        """
        self._video_path: str = video_path
        self._production: bool = production

        self._cap = cv2.VideoCapture(video_path)
        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self._total: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_w: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._scale: float = _DISPLAY_HEIGHT / self._vid_h
        self._disp_w: int = int(self._vid_w * self._scale)
        self._disp_h: int = _DISPLAY_HEIGHT
        self._frame_delay: int = max(1, int(1000 / self._fps))

        # Calibrate the coordinate mapper from a mid-game frame
        self._mapper = CoordinateMapper()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(self._total * 0.10)))
        ret, calib = self._cap.read()
        if ret:
            self._mapper.calibrate_from_frame(calib)
        else:
            self._mapper.calibrate_from_image(self._vid_w, self._vid_h)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        entries: List[RecommendationDict] = _load_jsonl(jsonl_path)
        self._timestamps: List[int] = [e["timestamp_ms"] for e in entries]
        self._entries: List[RecommendationDict] = entries
        self._rec_entries: List[RecommendationDict] = [
            entry for entry in entries if entry.get("has_recommendation")
        ]

        print(f"[VideoPlayer] {self._vid_w}x{self._vid_h} @ {self._fps:.1f} fps")
        print(
            f"[VideoPlayer] {len(entries)} frames, "
            f"{len(self._rec_entries)} with recommendations"
        )

    def _draw_overlay(
        self, frame: np.ndarray, entry: RecommendationDict
    ) -> np.ndarray:
        """
        Draw the card placement recommendation box on a display frame.

        Args:
            frame: the scaled display frame (will be modified in place).
            entry: the recommendation entry for this timestamp.

        Returns:
            The modified frame.
        """
        if not entry.get("has_recommendation"):
            return frame
        card: str = entry.get("card", "")
        tile_x: int = entry.get("tile_x", 0)
        tile_y: int = entry.get("tile_y", 0)
        elixir_req: int = entry.get("elixir_required", 0)

        if self._mapper.bounds is None:
            return frame

        box_x1, box_y1, box_x2, box_y2 = self._mapper.tile_bounds_pixels(tile_x, tile_y)
        scale = self._scale
        disp_x1 = int(box_x1 * scale)
        disp_y1 = int(box_y1 * scale)
        disp_x2 = int(box_x2 * scale)
        disp_y2 = int(box_y2 * scale)

        player_elixir: int = entry.get("player_elixir", 0) or 0
        dim = self._production and player_elixir < elixir_req
        color = (0, 140, 255) if not dim else (60, 60, 60)

        cv2.rectangle(frame, (disp_x1, disp_y1), (disp_x2, disp_y2), color, 2)
        label = card
        (text_w, text_h), _ = cv2.getTextSize(label, _FONT, 0.4, 1)
        cv2.rectangle(
            frame, (disp_x1, disp_y1 - text_h - 4), (disp_x1 + text_w + 4, disp_y1), color, -1
        )
        cv2.putText(frame, label, (disp_x1 + 2, disp_y1 - 2), _FONT, 0.4, (255, 255, 255), 1)
        return frame

    def _find_entry(self, ts_ms: int) -> Optional[RecommendationDict]:
        """
        Binary-search for the most recent entry at or before ts_ms.

        Args:
            ts_ms: current video timestamp in milliseconds.

        Returns:
            The matching entry, or None if the list is empty.
        """
        lo, hi = 0, len(self._timestamps) - 1
        result: Optional[RecommendationDict] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._timestamps[mid] <= ts_ms:
                result = self._entries[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def _status_bar(
        self, frame: np.ndarray, entry: Optional[RecommendationDict]
    ) -> np.ndarray:
        """
        Append a one-line status bar below the frame.

        Args:
            frame: display frame to append to.
            entry: current recommendation entry, or None.

        Returns:
            Stacked frame with status bar appended at the bottom.
        """
        _, frame_w = frame.shape[:2]
        status_strip = np.zeros((24, frame_w, 3), dtype=np.uint8)
        if entry:
            ts_secs = entry["timestamp_ms"] / 1000
            elx = entry.get("player_elixir", 0)
            card = entry.get("card", "") if entry.get("has_recommendation") else "—"
            text = (
                f"t={ts_secs:.1f}s  elixir={elx}  rec={card}"
                "  overlay=ON  Space:pause H:overlay Q:quit"
            )
        else:
            text = "Space:pause  H:overlay  Q:quit"
        cv2.putText(status_strip, text, (4, 16), _FONT, 0.38, (200, 200, 200), 1)
        return np.vstack([frame, status_strip])

    def run(self) -> None:
        """Open the replay window and enter the playback loop."""
        window = "TorchRoyale Replay"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, self._disp_w, self._disp_h + 24)

        show_overlay: bool = True
        paused: bool = False
        current_entry: Optional[RecommendationDict] = None
        frame: Optional[np.ndarray] = None

        while True:
            if not paused:
                ret, frame = self._cap.read()
                if not ret:
                    break
                ts_ms: int = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
                current_entry = self._find_entry(ts_ms)

            if frame is None:
                break

            disp = cv2.resize(frame, (self._disp_w, self._disp_h))
            if show_overlay and current_entry:
                disp = self._draw_overlay(disp, current_entry)

            # Print hand / field info to console when overlay is active
            if current_entry:
                dets: List[Dict] = current_entry.get("detections", [])
                in_hand = [
                    det["class_name"].replace("-in-hand", "")
                    for det in dets if "-in-hand" in det.get("class_name", "")
                ]
                on_field_player = [
                    f"{det['class_name']}@({det['tile_x']},{det['tile_y']})"
                    for det in dets if det.get("is_on_field") and not det.get("is_opponent")
                ]
                on_field_opp = [
                    f"{det['class_name']}@({det['tile_x']},{det['tile_y']})"
                    for det in dets if det.get("is_on_field") and det.get("is_opponent")
                ]
                ts_secs = current_entry["timestamp_ms"] / 1000
                if in_hand:
                    opp_str = f" | Opp field: {', '.join(on_field_opp)}" if on_field_opp else ""
                    print(
                        f"[{ts_secs:6.1f}s] Hand: {', '.join(in_hand)}"
                        f" | Player field: {', '.join(on_field_player)}{opp_str}"
                    )

            disp = self._status_bar(disp, current_entry)
            cv2.imshow(window, disp)

            key = cv2.waitKey(self._frame_delay if not paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("k")):
                paused = not paused
            elif key == ord("h"):
                show_overlay = not show_overlay

        self._cap.release()
        cv2.destroyAllWindows()
        print("[VideoPlayer] Done.")

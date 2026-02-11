"""
Tower health detector using OCR to read HP numbers.

Detection logic:
- Towers display their current HP as white text on a health bar
- Player tower HP appears below the tower center
- Opponent tower HP appears above the tower center
- King tower: no HP text visible = tower is at full health
- Princess tower: always displays HP (even at full health). If not detected
  by Roboflow, the tower is destroyed.
- Tower level is read from the king tower to look up max HP
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from ..constants.game_constants import get_tower_max_hp


@dataclass
class TowerHealthResult:
    """Result of tower health detection."""
    hp_current: Optional[int]  # None if not detected
    hp_max: int
    health_percent: Optional[float]  # 0-100, None if unknown (OCR failure on princess)
    detected: bool  # Whether HP text was found
    is_destroyed: bool = False
    raw_text: str = ""


# Keep old class for backwards compatibility
@dataclass
class HealthBarResult:
    """Result of health bar detection (legacy)."""
    health_percent: float  # 0-100
    bar_visible: bool
    confidence: float


class TowerHealthDetector:
    """
    Detects tower HP by reading the number displayed on each tower.

    Uses Roboflow tower detections to locate towers, then runs OCR
    on the HP text region relative to each tower's bounding box.

    Usage:
        detector = TowerHealthDetector()
        results = detector.detect_all_towers(image, tower_detections, level=15)
    """

    def __init__(self):
        """Initialize the tower health detector."""
        from ..ocr.digit_detector import DigitDetector
        self._digit_detector = DigitDetector()

    def detect_tower_hp(
        self,
        image: np.ndarray,
        tower_cx: int,
        tower_cy: int,
        tower_w: int,
        tower_h: int,
        is_opponent: bool,
        is_king: bool,
        level: int = 15,
    ) -> TowerHealthResult:
        """
        Detect HP for a single tower using OCR.

        Args:
            image: Full frame (BGR format)
            tower_cx: Tower bounding box center x
            tower_cy: Tower bounding box center y
            tower_w: Tower bounding box width
            tower_h: Tower bounding box height
            is_opponent: True if opponent tower
            is_king: True if king tower
            level: Tower level for max HP lookup

        Returns:
            TowerHealthResult with current and max HP
        """
        img_h, img_w = image.shape[:2]
        hp_max = get_tower_max_hp(level, is_king)

        # Compute HP text search region relative to tower bbox
        hp_region = self._get_hp_region(
            tower_cx, tower_cy, tower_w, tower_h,
            is_opponent, img_w, img_h, is_king=is_king
        )

        if hp_region is None:
            # King tower: assume full HP. Princess tower: unknown (OCR failure).
            if is_king:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=100.0, detected=False
                )
            else:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=None, detected=False
                )

        x1, y1, x2, y2 = hp_region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            if is_king:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=100.0, detected=False
                )
            else:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=None, detected=False
                )

        # Preprocess and run OCR
        processed = self._digit_detector._preprocess_for_ocr(roi)
        try:
            results = self._digit_detector.reader.readtext(
                processed,
                allowlist='0123456789',
                paragraph=False,
                min_size=5,
            )
        except Exception:
            if is_king:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=100.0, detected=False
                )
            else:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=None, detected=False
                )

        if not results:
            # King tower: no text = full HP. Princess tower: no text = OCR failure.
            if is_king:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=100.0, detected=False
                )
            else:
                return TowerHealthResult(
                    hp_current=None, hp_max=hp_max,
                    health_percent=None, detected=False
                )

        # Try each OCR result (sorted by confidence) to find a valid HP
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        for bbox, text, confidence in sorted_results:
            raw_text = text.strip()
            hp_value = self._parse_hp_text(raw_text, hp_max)
            if hp_value is not None and confidence > 0.3:
                health_pct = min(100.0, (hp_value / hp_max) * 100)
                return TowerHealthResult(
                    hp_current=hp_value, hp_max=hp_max,
                    health_percent=health_pct, detected=True,
                    raw_text=raw_text
                )

        # No valid HP parsed from OCR results
        all_text = "|".join(r[1] for r in results)
        if is_king:
            return TowerHealthResult(
                hp_current=None, hp_max=hp_max,
                health_percent=100.0, detected=False,
                raw_text=all_text
            )
        else:
            return TowerHealthResult(
                hp_current=None, hp_max=hp_max,
                health_percent=None, detected=False,
                raw_text=all_text
            )

    def _get_hp_region(
        self,
        cx: int, cy: int, bw: int, bh: int,
        is_opponent: bool,
        img_w: int, img_h: int,
        is_king: bool = False,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute the pixel region where HP text appears relative to tower bbox.

        The HP number sits on a health bar next to a level badge.
        - Player princess towers: HP bar is near tower center
        - Player king tower: HP bar is further below center (taller tower)
        - Opponent towers: HP bar is above the tower (near top of bbox)

        Returns:
            (x1, y1, x2, y2) or None
        """
        # Wide enough to capture the full HP number
        x1 = max(0, cx - int(bw * 0.4))
        x2 = min(img_w, cx + int(bw * 0.9))

        if is_opponent and is_king:
            # Opponent king: HP bar just above the tower (king bbox is tall)
            y1 = max(0, cy - int(bh * 0.4))
            y2 = max(0, cy - int(bh * 0.2))
        elif is_opponent:
            # Opponent princess: HP on pink bar above the tower
            y1 = max(0, cy - int(bh * 0.6))
            y2 = max(0, cy - int(bh * 0.25))
        elif is_king:
            # Player king tower: HP is near the bottom of the tall bbox
            y1 = min(img_h, cy + int(bh * 0.2))
            y2 = min(img_h, cy + int(bh * 0.5))
        else:
            # Player princess towers: HP near tower center
            y1 = max(0, cy - int(bh * 0.1))
            y2 = min(img_h, cy + int(bh * 0.25))

        if y2 <= y1 or x2 <= x1:
            return None

        return (x1, y1, x2, y2)

    def _parse_hp_text(self, text: str, hp_max: int) -> Optional[int]:
        """
        Parse OCR text into an HP value.

        OCR often reads the level badge (1-2 digits) concatenated with the
        HP text (3-5 digits), e.g. "152384" = level 15 + HP 2384.
        We try progressively shorter suffixes to find a valid HP value.

        Args:
            text: Raw OCR text (digits only)
            hp_max: Maximum possible HP for this tower

        Returns:
            HP value or None if invalid
        """
        if not text or not text.isdigit():
            return None

        # Try substrings from the end: last 5, 4, 3 digits
        # This strips the level badge prefix
        for length in range(min(5, len(text)), 2, -1):
            suffix = text[-length:]
            value = int(suffix)
            if 100 <= value <= hp_max:
                return value

        # Accept small HP values (1-99) when text is short (1-2 chars).
        # These can't be confused with a level badge prefix since
        # the OCR search region only covers where HP text appears.
        if len(text) <= 2:
            value = int(text)
            if 1 <= value <= hp_max:
                return value

        return None

    def detect_tower_level(
        self,
        image: np.ndarray,
        king_cx: int,
        king_cy: int,
        king_w: int,
        king_h: int,
        is_opponent: bool,
    ) -> int:
        """
        Detect tower level from the king tower's golden crown badge.

        Args:
            image: Full frame (BGR format)
            king_cx, king_cy: King tower bbox center
            king_w, king_h: King tower bbox dimensions
            is_opponent: True if opponent king tower

        Returns:
            Detected level (1-16), defaults to 15 if detection fails
        """
        img_h, img_w = image.shape[:2]

        # The level badge is near the bottom-left of player king tower
        # or top-left of opponent king tower
        badge_w = king_w // 3
        badge_h = king_h // 4

        if is_opponent:
            x1 = max(0, king_cx - king_w // 2 - badge_w // 2)
            y1 = max(0, king_cy - king_h // 2 - badge_h)
            x2 = min(img_w, x1 + badge_w)
            y2 = min(img_h, y1 + badge_h)
        else:
            x1 = max(0, king_cx - king_w // 2 - badge_w // 2)
            y1 = min(img_h, king_cy + king_h // 4)
            x2 = min(img_w, x1 + badge_w)
            y2 = min(img_h, y1 + badge_h)

        if y2 <= y1 or x2 <= x1:
            return 15

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return 15

        processed = self._digit_detector._preprocess_for_ocr(roi)
        try:
            results = self._digit_detector.reader.readtext(
                processed,
                allowlist='0123456789',
                paragraph=False,
                min_size=5,
            )
        except Exception:
            return 15

        if not results:
            return 15

        best = max(results, key=lambda x: x[2])
        text = best[1].strip()

        if text.isdigit():
            level = int(text)
            if 1 <= level <= 16:
                return level

        return 15

    def detect_all_towers(
        self,
        image: np.ndarray,
        tower_detections: List[Dict],
        player_level: int = 15,
        opponent_level: int = 15,
    ) -> Dict[str, TowerHealthResult]:
        """
        Detect HP for all towers from Roboflow detections.

        Princess towers that are NOT detected by Roboflow are considered
        destroyed (Roboflow only fails to detect them when they're gone).

        Args:
            image: Full frame (BGR format)
            tower_detections: List of dicts with keys:
                class_name, pixel_x, pixel_y, pixel_width, pixel_height, is_opponent
            player_level: Player tower level
            opponent_level: Opponent tower level

        Returns:
            Dict mapping tower names to TowerHealthResult
        """
        results = {}

        for det in tower_detections:
            class_name = det["class_name"]
            is_opponent = det["is_opponent"]
            is_king = "king" in class_name

            # Determine tower name for the results dict
            if is_king:
                name = "opponent_king" if is_opponent else "player_king"
            else:
                # Distinguish left vs right princess tower by x position
                img_center_x = image.shape[1] // 2
                side = "left" if det["pixel_x"] < img_center_x else "right"
                prefix = "opponent" if is_opponent else "player"
                name = f"{prefix}_{side}"

            level = opponent_level if is_opponent else player_level

            results[name] = self.detect_tower_hp(
                image,
                tower_cx=det["pixel_x"],
                tower_cy=det["pixel_y"],
                tower_w=det["pixel_width"],
                tower_h=det["pixel_height"],
                is_opponent=is_opponent,
                is_king=is_king,
                level=level,
            )

        # Princess towers not detected by Roboflow = destroyed
        princess_tower_names = [
            "player_left", "player_right",
            "opponent_left", "opponent_right",
        ]
        for name in princess_tower_names:
            if name not in results:
                is_opp = name.startswith("opponent")
                level = opponent_level if is_opp else player_level
                hp_max = get_tower_max_hp(level, is_king=False)
                results[name] = TowerHealthResult(
                    hp_current=0, hp_max=hp_max,
                    health_percent=0.0, detected=True,
                    is_destroyed=True,
                )

        return results

    def get_hp_region(
        self,
        tower_cx: int,
        tower_cy: int,
        tower_w: int,
        tower_h: int,
        is_opponent: bool,
        img_w: int,
        img_h: int,
        is_king: bool = False,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Public access to HP region computation for debug drawing."""
        return self._get_hp_region(
            tower_cx, tower_cy, tower_w, tower_h,
            is_opponent, img_w, img_h, is_king=is_king
        )

    def __repr__(self) -> str:
        return "TowerHealthDetector()"


# Keep old class available for backwards compatibility
class HealthBarDetector:
    """Legacy health bar detector. Use TowerHealthDetector instead."""

    def detect_health(self, image, region) -> HealthBarResult:
        return HealthBarResult(health_percent=100.0, bar_visible=False, confidence=0.0)

    def __repr__(self) -> str:
        return "HealthBarDetector(legacy)"

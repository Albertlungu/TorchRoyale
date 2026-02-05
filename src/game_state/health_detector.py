"""
Health bar detector for towers and troops.

Detection logic:
- No health bar visible = 100% health (full HP)
- Health bar visible = (colored portion / total bar length) * 100

Health bars use color gradients:
- Green: High health
- Yellow/Orange: Medium health
- Red: Low health
"""

import cv2
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class HealthBarResult:
    """Result of health bar detection."""
    health_percent: float  # 0-100
    bar_visible: bool
    confidence: float


class HealthBarDetector:
    """
    Detects health percentage from health bars.

    Uses color-based detection to identify:
    - Health bar presence
    - Colored (remaining health) vs gray (lost health) portions

    Tower health bars appear above destroyed portions,
    troop health bars appear above the unit.
    """

    def __init__(self):
        """Initialize health bar detector with color ranges."""
        # Health bar colors (HSV ranges)
        # Green: High health
        self.green_lower = np.array([35, 80, 80])
        self.green_upper = np.array([85, 255, 255])

        # Yellow/Orange: Medium health
        self.yellow_lower = np.array([15, 80, 80])
        self.yellow_upper = np.array([35, 255, 255])

        # Red: Low health
        self.red_lower1 = np.array([0, 80, 80])
        self.red_upper1 = np.array([15, 255, 255])
        self.red_lower2 = np.array([165, 80, 80])
        self.red_upper2 = np.array([180, 255, 255])

        # Gray background (depleted health)
        self.gray_lower = np.array([0, 0, 40])
        self.gray_upper = np.array([180, 50, 150])

    def detect_health(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> HealthBarResult:
        """
        Detect health percentage from a region.

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max) of health bar area

        Returns:
            HealthBarResult with health percentage
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return HealthBarResult(
                health_percent=100.0,
                bar_visible=False,
                confidence=0.0
            )

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create masks for health colors
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Red needs two ranges (wraps around in HSV)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Combine all health colors
        health_mask = cv2.bitwise_or(green_mask, yellow_mask)
        health_mask = cv2.bitwise_or(health_mask, red_mask)

        # Gray background (depleted portion)
        gray_mask = cv2.inRange(hsv, self.gray_lower, self.gray_upper)

        # Total bar area
        total_mask = cv2.bitwise_or(health_mask, gray_mask)

        # Count pixels
        health_pixels = cv2.countNonZero(health_mask)
        total_pixels = cv2.countNonZero(total_mask)

        # Minimum pixels to consider bar present
        min_bar_pixels = roi.shape[0] * roi.shape[1] * 0.02  # 2% of region

        if total_pixels < min_bar_pixels:
            # No health bar detected - assume full health
            return HealthBarResult(
                health_percent=100.0,
                bar_visible=False,
                confidence=0.9
            )

        # Calculate health percentage
        if total_pixels > 0:
            health_percent = (health_pixels / total_pixels) * 100
        else:
            health_percent = 100.0

        # Determine confidence based on detection quality
        confidence = min(1.0, total_pixels / (min_bar_pixels * 5))

        return HealthBarResult(
            health_percent=min(100.0, max(0.0, health_percent)),
            bar_visible=True,
            confidence=confidence
        )

    def detect_health_horizontal(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> HealthBarResult:
        """
        Detect health from a horizontal bar using left-to-right analysis.

        Better for tower health bars where colored portion extends from left.

        Args:
            image: Full frame (BGR format)
            region: (x_min, y_min, x_max, y_max)

        Returns:
            HealthBarResult
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]

        if roi.size == 0 or roi.shape[1] == 0:
            return HealthBarResult(
                health_percent=100.0,
                bar_visible=False,
                confidence=0.0
            )

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Combine all health colors
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)

        health_mask = cv2.bitwise_or(green_mask, yellow_mask)
        health_mask = cv2.bitwise_or(health_mask, red_mask1)
        health_mask = cv2.bitwise_or(health_mask, red_mask2)

        # Find rightmost health pixel (edge of remaining health)
        column_sums = np.sum(health_mask, axis=0)

        if np.max(column_sums) < roi.shape[0] * 0.1:  # Less than 10% of height
            return HealthBarResult(
                health_percent=100.0,
                bar_visible=False,
                confidence=0.9
            )

        # Find the rightmost column with significant health pixels
        threshold = roi.shape[0] * 0.2
        health_columns = np.where(column_sums > threshold)[0]

        if len(health_columns) == 0:
            return HealthBarResult(
                health_percent=0.0,
                bar_visible=True,
                confidence=0.7
            )

        rightmost_health = health_columns[-1]
        total_width = roi.shape[1]

        health_percent = (rightmost_health / total_width) * 100

        return HealthBarResult(
            health_percent=min(100.0, max(0.0, health_percent)),
            bar_visible=True,
            confidence=0.8
        )

    def detect_all_towers(
        self,
        image: np.ndarray,
        tower_regions: Dict[str, Tuple[int, int, int, int]]
    ) -> Dict[str, HealthBarResult]:
        """
        Detect health for all towers.

        Args:
            image: Full frame
            tower_regions: Dict mapping tower names to region tuples

        Returns:
            Dict mapping tower names to HealthBarResult
        """
        results = {}

        for tower_name, region in tower_regions.items():
            # Use horizontal detection for towers
            results[tower_name] = self.detect_health_horizontal(image, region)

        return results

    def detect_troop_health(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> HealthBarResult:
        """
        Detect health for a troop from its bounding box.

        Troop health bars appear above the unit, so we look
        in the area above the bounding box.

        Args:
            image: Full frame
            bbox: (x_min, y_min, x_max, y_max) of the troop

        Returns:
            HealthBarResult
        """
        x1, y1, x2, y2 = bbox

        # Health bar is typically above the unit
        bar_height = int((y2 - y1) * 0.15)  # ~15% of unit height
        bar_y1 = max(0, y1 - bar_height)
        bar_y2 = y1

        # Slightly wider than unit
        padding = int((x2 - x1) * 0.1)
        bar_x1 = max(0, x1 - padding)
        bar_x2 = min(image.shape[1], x2 + padding)

        return self.detect_health_horizontal(image, (bar_x1, bar_y1, bar_x2, bar_y2))

    def __repr__(self) -> str:
        return "HealthBarDetector()"

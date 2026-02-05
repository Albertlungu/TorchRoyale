"""
TorchRoyale Detection Test

This script demonstrates the full detection pipeline:
1. Load an image from the test data
2. Run Roboflow object detection
3. Convert pixel coordinates to tile grid coordinates
4. Display results with annotations and grid mapping

Usage:
    python detection_test.py [image_path]

If no image path is provided, uses the first image in tests/data/
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import cv2
import supervision as sv
import cv2
import mss
import numpy as np
import supervision as sv
from inference import get_model

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from grid.coordinate_mapper import CoordinateMapper, ArenaBounds, print_grid
from grid.validity_masks import PlacementValidator, CARD_TYPES, ON_FIELD_CARDS


# Configuration
ROBOFLOW_MODEL_ID = "torchroyale/4"
DEFAULT_TEST_DATA_DIR = Path("tests/data")


@dataclass
class Detection:
    """Represents a single detected object with both pixel and tile coordinates."""
    class_name: str
    confidence: float
    # Pixel coordinates (from Roboflow)
    pixel_x: int
    pixel_y: int
    pixel_width: int
    pixel_height: int
    # Tile coordinates (mapped)
    tile_x: int
    tile_y: int
    # Metadata
    is_opponent: bool
    is_on_field: bool

    def __repr__(self) -> str:
        side = "opponent" if self.is_opponent else "friendly"
        status = "on-field" if self.is_on_field else "in-hand"
        return (f"Detection({self.class_name}, tile=({self.tile_x}, {self.tile_y}), "
                f"conf={self.confidence:.2f}, {side}, {status})")


class DetectionPipeline:
    """
    Complete detection pipeline from image to grid-mapped detections.
    """

    def __init__(self, model_id: str = ROBOFLOW_MODEL_ID):
        """
        Initialize the detection pipeline.

        Args:
            model_id: Roboflow model identifier
        """
        print(f"Loading Roboflow model: {model_id}")
        self.model = get_model(model_id=model_id)
        self.mapper: Optional[CoordinateMapper] = None
        self.validator: Optional[PlacementValidator] = None

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the full pipeline.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
            - detections: List of Detection objects
            - image: Original image (numpy array)
            - annotated_image: Image with bounding boxes drawn
            - grid_state: 2D representation of what's on each tile
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        height, width = image.shape[:2]
        print(f"Image dimensions: {width}x{height}")

        # Calibrate coordinate mapper for this image
        self._calibrate_mapper(width, height)

        # Run Roboflow detection
        print("Running Roboflow inference...")
        results = self.model.infer(image)[0]

        # Convert to supervision format for visualization
        sv_detections = sv.Detections.from_inference(results)

        # Extract and map detections
        detections = self._extract_detections(results)

        # Create annotated image
        annotated_image = self._annotate_image(image.copy(), sv_detections, detections)

        # Build grid state
        grid_state = self._build_grid_state(detections)

        return {
            "detections": detections,
            "image": image,
            "annotated_image": annotated_image,
            "grid_state": grid_state,
            "mapper": self.mapper,
        }

    def _calibrate_mapper(self, image_width: int, image_height: int):
        """
        Calibrate the coordinate mapper for the image dimensions.

        Adjust the arena_top_ratio and arena_bottom_ratio based on your
        screen capture setup. These defaults work for standard mobile captures.
        """
        self.mapper = CoordinateMapper()
        self.mapper.calibrate_from_image(
            image_width=image_width,
            image_height=image_height,
            arena_top_ratio=0.10,   # Arena starts 10% from top
            arena_bottom_ratio=0.80  # Arena ends 80% from top (above card hand)
        )
        self.validator = PlacementValidator(self.mapper)

        print(f"Coordinate mapper calibrated: {self.mapper}")
        print(f"Arena bounds: x=[{self.mapper.bounds.x_min}, {self.mapper.bounds.x_max}], "
              f"y=[{self.mapper.bounds.y_min}, {self.mapper.bounds.y_max}]")

    def _extract_detections(self, results: Any) -> List[Detection]:
        """
        Extract detections from Roboflow results and map to tile coordinates.

        Args:
            results: Raw Roboflow inference results

        Returns:
            List of Detection objects with tile coordinates
        """
        detections = []

        for prediction in results.predictions:
            # Get pixel coordinates (center of bounding box)
            pixel_x = int(prediction.x)
            pixel_y = int(prediction.y)
            pixel_width = int(prediction.width)
            pixel_height = int(prediction.height)

            # Map to tile coordinates
            tile_x, tile_y = self.mapper.pixel_to_tile(pixel_x, pixel_y)

            # Parse class name for metadata
            class_name = prediction.class_name
            is_opponent = class_name.startswith("opponent-")
            is_on_field = class_name in ON_FIELD_CARDS or "-on-field" in class_name.lower()

            # Clean class name (remove opponent- prefix for consistency)
            clean_name = class_name.replace("opponent-", "")

            detection = Detection(
                class_name=clean_name,
                confidence=prediction.confidence,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                tile_x=tile_x,
                tile_y=tile_y,
                is_opponent=is_opponent,
                is_on_field=is_on_field,
            )
            detections.append(detection)

        return detections

    def _annotate_image(self, image, sv_detections, detections: List[Detection]):
        """
        Annotate image with bounding boxes and tile coordinates.
        """
        # Use supervision for bounding boxes
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Create labels with tile coordinates
        labels = []
        for det in detections:
            side = "OPP" if det.is_opponent else "YOU"
            label = f"{det.class_name[:10]} ({det.tile_x},{det.tile_y}) [{side}]"
            labels.append(label)

        # Annotate
        annotated = box_annotator.annotate(scene=image, detections=sv_detections)
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=sv_detections,
            labels=labels
        )

        # Draw grid overlay (optional - shows tile boundaries)
        annotated = self._draw_grid_overlay(annotated)

        return annotated

    def _draw_grid_overlay(self, image, alpha: float = 0.3):
        """
        Draw a semi-transparent grid overlay on the image.
        """
        overlay = image.copy()

        # Draw vertical lines
        for x in range(self.mapper.GRID_WIDTH + 1):
            px = int(x * self.mapper.tile_width + self.mapper.bounds.x_min)
            cv2.line(overlay,
                     (px, self.mapper.bounds.y_min),
                     (px, self.mapper.bounds.y_max),
                     (255, 255, 255), 1)

        # Draw horizontal lines
        for y in range(self.mapper.GRID_HEIGHT + 1):
            py = int(y * self.mapper.tile_height + self.mapper.bounds.y_min)
            cv2.line(overlay,
                     (self.mapper.bounds.x_min, py),
                     (self.mapper.bounds.x_max, py),
                     (255, 255, 255), 1)

        # Highlight river tiles
        for y in self.mapper.RIVER_ROWS:
            for x in range(self.mapper.GRID_WIDTH):
                if not self.mapper.is_bridge(x, y):
                    x1, y1, x2, y2 = self.mapper.get_tile_bounds_pixels(x, y)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 100), -1)

        # Blend overlay with original
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def _build_grid_state(self, detections: List[Detection]) -> List[List[Optional[str]]]:
        """
        Build a 2D grid representation of the game state.

        Returns:
            32x18 grid where each cell contains the class name of the unit there,
            or None if empty.
        """
        grid = [[None for _ in range(self.mapper.GRID_WIDTH)]
                for _ in range(self.mapper.GRID_HEIGHT)]

        for det in detections:
            if det.is_on_field:
                prefix = "opp:" if det.is_opponent else ""
                grid[det.tile_y][det.tile_x] = f"{prefix}{det.class_name}"

        return grid


def print_detections(detections: List[Detection]):
    """Print detection summary."""
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)

    friendly = [d for d in detections if not d.is_opponent]
    opponent = [d for d in detections if d.is_opponent]

    print(f"\nFriendly units ({len(friendly)}):")
    for det in friendly:
        status = "on-field" if det.is_on_field else "in-hand"
        print(f"  - {det.class_name:20s} @ tile ({det.tile_x:2d}, {det.tile_y:2d}) "
              f"| pixel ({det.pixel_x:4d}, {det.pixel_y:4d}) | conf: {det.confidence:.2f} | {status}")

    print(f"\nOpponent units ({len(opponent)}):")
    for det in opponent:
        status = "on-field" if det.is_on_field else "in-hand"
        print(f"  - {det.class_name:20s} @ tile ({det.tile_x:2d}, {det.tile_y:2d}) "
              f"| pixel ({det.pixel_x:4d}, {det.pixel_y:4d}) | conf: {det.confidence:.2f} | {status}")


def print_grid_state(grid_state: List[List[Optional[str]]], mapper: CoordinateMapper):
    """Print a text visualization of the grid state."""
    print("\n" + "=" * 60)
    print("GRID STATE (what's on each tile)")
    print("=" * 60)
    print("Legend: . = empty, ~ = river, B = bridge, [X] = unit")
    print("-" * (mapper.GRID_WIDTH * 2 + 4))

    for y in range(mapper.GRID_HEIGHT):
        row = f"{y:2d} |"
        for x in range(mapper.GRID_WIDTH):
            cell = grid_state[y][x]
            if cell:
                # Show first letter of unit
                if cell.startswith("opp:"):
                    row += cell[4].upper()  # Opponent units uppercase
                else:
                    row += cell[0].lower()  # Friendly units lowercase
            elif mapper.is_bridge(x, y):
                row += "B"
            elif mapper.is_river(x, y):
                row += "~"
            else:
                row += "."
        row += "|"
        print(row)

    print("-" * (mapper.GRID_WIDTH * 2 + 4))


def get_default_test_image() -> str:
    """Get the first available test image."""
    test_images = sorted(DEFAULT_TEST_DATA_DIR.glob("*.png"))
    if not test_images:
        raise FileNotFoundError(f"No PNG images found in {DEFAULT_TEST_DATA_DIR}")
    return str(test_images[0])


def main():
    """Main entry point."""
    # Get image path from argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = get_default_test_image()
        print(f"Using default test image: {image_path}")

    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Initialize pipeline
    pipeline = DetectionPipeline()

    # Process image
    results = pipeline.process_image(image_path)

    # Print results
    print_detections(results["detections"])
    print_grid_state(results["grid_state"], results["mapper"])

    # Show coordinate mapping info
    print("\n" + "=" * 60)
    print("COORDINATE MAPPER INFO")
    print("=" * 60)
    print(results["mapper"].to_dict())

    # Display annotated image
    print("\nDisplaying annotated image (press any key to close)...")
    sv.plot_image(results["annotated_image"])

    # Optionally save the annotated image
    output_path = Path(image_path).stem + "_annotated.png"
    cv2.imwrite(output_path, results["annotated_image"])
    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()

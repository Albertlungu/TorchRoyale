"""Live state visualization for TorchRoyale."""

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal

from src.namespaces.numbers import NumberDetection
from src.namespaces.state import State
from src.namespaces.units import NAME2UNIT


REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_DIR = REPO_ROOT / "debug"
SCREENSHOTS_DIR = DEBUG_DIR / "screenshots"
LABELS_DIR = DEBUG_DIR / "labels"
CARD_CONFIG = [
    (21, 609, 47, 642),
    (84, 543, 145, 616),
    (153, 543, 214, 616),
    (222, 543, 283, 616),
    (291, 543, 352, 616),
]


class Visualizer(QObject):
    """Annotate live detector state and optionally emit it to the UI."""

    _COLOUR_AND_RGBA = [
        (0, 38, 63, 127),
        (0, 120, 210, 127),
        (115, 221, 252, 127),
        (15, 205, 202, 127),
        (52, 153, 114, 127),
        (0, 204, 84, 127),
        (1, 255, 127, 127),
        (255, 216, 70, 127),
        (255, 125, 57, 127),
        (255, 47, 65, 127),
        (135, 13, 75, 127),
        (246, 0, 184, 127),
        (179, 17, 193, 127),
        (168, 168, 168, 127),
        (220, 220, 220, 127),
    ]

    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, save_labels: bool, save_images: bool, show_images: bool) -> None:
        super().__init__()
        self.save_labels = save_labels
        self.save_images = save_images
        self.show_images = show_images
        self.font = ImageFont.load_default()
        self.unit_names = [unit["name"] for unit in NAME2UNIT.values()]
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_label(image: Image.Image, state: State, basename: int) -> None:
        labels = []
        for detection in state.allies + state.enemies:
            bbox = detection.position.bbox
            xc = (bbox[0] + bbox[2]) / (2 * image.width)
            yc = (bbox[1] + bbox[3]) / (2 * image.height)
            w = (bbox[2] - bbox[0]) / image.width
            h = (bbox[3] - bbox[1]) / image.height
            labels.append(f"{detection.unit.name} {xc} {yc} {w} {h}")
        (LABELS_DIR / f"{basename}.txt").write_text("\n".join(labels), encoding="utf-8")

    def _draw_text(
        self,
        drawing: ImageDraw.ImageDraw,
        bbox: tuple[int, int, int, int],
        text: str,
        rgba: tuple[int, int, int, int] = (0, 0, 0, 255),
    ) -> None:
        text_width, text_height = drawing.textbbox((0, 0), text, font=self.font)[2:]
        text_bbox = (
            bbox[0],
            bbox[1] - text_height,
            bbox[0] + text_width,
            bbox[1],
        )
        drawing.rectangle(text_bbox, fill=rgba)
        drawing.rectangle(bbox, outline=rgba)
        drawing.text(text_bbox[:2], text=text, fill="white", font=self.font)

    def _draw_unit_bboxes(
        self,
        drawing: ImageDraw.ImageDraw,
        detections,
        prefix: str,
    ) -> None:
        for detection in detections:
            colour_index = self.unit_names.index(detection.unit.name) % len(
                self._COLOUR_AND_RGBA
            )
            rgba = self._COLOUR_AND_RGBA[colour_index]
            self._draw_text(
                drawing,
                detection.position.bbox,
                f"{prefix}_{detection.unit.name}",
                rgba,
            )

    def _annotate_image(self, image: Image.Image, state: State) -> Image.Image:
        drawing = ImageDraw.Draw(image, "RGBA")
        for payload in asdict(state.numbers).values():
            detection = NumberDetection(**payload)
            drawing.rectangle(detection.bbox)
            self._draw_text(drawing, detection.bbox, f"{detection.number:.2f}")

        self._draw_unit_bboxes(drawing, state.allies, "ally")
        self._draw_unit_bboxes(drawing, state.enemies, "enemy")

        for card, position in zip(state.cards, CARD_CONFIG):
            drawing.rectangle(position)
            self._draw_text(drawing, position, card.name)

        return image

    def run(self, image: Image.Image, state: State) -> None:
        screenshot_count = len(list(SCREENSHOTS_DIR.glob("*.png")))
        label_count = len(list(LABELS_DIR.glob("*.txt")))
        basename = max(screenshot_count, label_count) + 1

        if self.save_labels:
            self._write_label(image, state, basename)

        if not self.save_images and not self.show_images:
            return

        annotated = self._annotate_image(image.copy(), state)
        if self.save_images:
            annotated.save(SCREENSHOTS_DIR / f"{basename}.png")
        if self.show_images:
            self.frame_ready.emit(np.array(annotated))

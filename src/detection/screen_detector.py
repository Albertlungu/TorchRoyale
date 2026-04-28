"""Screen detector based on reference screen crops."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.namespaces import Screen
from src.namespaces import Screens

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREEN_IMAGES_DIR = REPO_ROOT / "data" / "images" / "screen"


class ScreenDetector:
    """Detect the current coarse game screen."""

    def __init__(self, hash_size: int = 8, threshold: float = 30) -> None:
        self.hash_size = hash_size
        self.threshold = threshold
        self.screen_hashes = self._calculate_screen_hashes()

    def _image_hash(self, image: Image.Image) -> np.ndarray:
        crop = image.resize(
            (self.hash_size, self.hash_size), Image.Resampling.BILINEAR
        )
        return np.array(crop, dtype=np.float32).flatten()

    def _calculate_screen_hashes(self) -> dict[Screen, np.ndarray]:
        screen_hashes: dict[Screen, np.ndarray] = {}
        for screen in Screens.__dict__.values():
            if not isinstance(screen, Screen) or screen.ltrb is None:
                continue
            path = SCREEN_IMAGES_DIR / f"{screen.name}.jpg"
            if not path.exists():
                raise FileNotFoundError(f"Missing screen reference image: {path}")
            image = Image.open(path)
            screen_hashes[screen] = self._image_hash(image)
        return screen_hashes

    def run(self, image: Image.Image) -> Screen:
        current_screen = Screens.UNKNOWN
        best_diff = self.threshold

        for screen in Screens.__dict__.values():
            if not isinstance(screen, Screen) or screen.ltrb is None:
                continue
            treated_ltrb = (
                int(screen.ltrb[0] * image.size[0] / 720),
                int(screen.ltrb[1] * image.size[1] / 1280),
                int(screen.ltrb[2] * image.size[0] / 720),
                int(screen.ltrb[3] * image.size[1] / 1280),
            )
            hash_ = self._image_hash(image.crop(treated_ltrb))
            target_hash = self.screen_hashes[screen]
            diff = np.mean(np.abs(hash_ - target_hash))
            if diff < best_diff:
                best_diff = diff
                current_screen = screen

        return current_screen

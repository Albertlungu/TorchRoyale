"""Side classifier for distinguishing ally and enemy units."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.detection.onnx_detector import OnnxDetector


class SideDetector(OnnxDetector):
    """Classify whether a detected unit belongs to the ally or enemy side."""

    SIDE_SIZE = 16

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(
            (self.SIDE_SIZE, self.SIDE_SIZE), Image.Resampling.BICUBIC
        )
        array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def run(self, image: Image.Image) -> str:
        prediction = self._infer(self._preprocess(image))
        return ("ally", "enemy")[int(np.argmax(prediction[0]))]

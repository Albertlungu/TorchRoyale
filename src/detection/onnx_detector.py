"""Base utilities for ONNX-backed detectors."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


class OnnxDetector:
    """Shared ONNX runtime wrapper."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        providers = list(
            set(ort.get_available_providers())
            & {"CUDAExecutionProvider", "CPUExecutionProvider"}
        )
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.output_name = self.sess.get_outputs()[0].name
        input_ = self.sess.get_inputs()[0]
        self.input_name = input_.name
        self.model_height, self.model_width = input_.shape[2:]

    def resize(self, image) -> np.ndarray:
        import PIL.Image

        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        ratio = image.height / image.width
        if ratio > self.model_height / self.model_width:
            height = self.model_height
            width = int(self.model_height / ratio)
        else:
            width = self.model_width
            height = int(self.model_width * ratio)
        return np.array(image.resize((width, height)))

    def pad(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        dx = self.model_width - width
        dy = self.model_height - height
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=114,
        )
        return padded, padding

    def resize_pad_transpose_and_scale(
        self,
        image,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        image = self.resize(image)
        image = np.array(image, dtype=np.float16)
        image, padding = self.pad(image)
        image = image.transpose(2, 0, 1)
        image /= 255.0
        return image, padding

    def fix_bboxes(
        self,
        boxes: np.ndarray,
        width: int,
        height: int,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[2]
        boxes[..., [0, 2]] *= width / (self.model_width - padding[0] - padding[1])
        boxes[..., [1, 3]] *= height / (
            self.model_height - padding[2] - padding[3]
        )
        return boxes

    def _infer(self, image: np.ndarray) -> np.ndarray:
        return self.sess.run([self.output_name], {self.input_name: image})[0]

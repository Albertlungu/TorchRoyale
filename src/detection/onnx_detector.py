"""
Base ONNX detector for TorchRoyale.

Provides common functionality for loading and running ONNX models
for game object detection.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


class OnnxDetector:
    """Base class for ONNX model inference.

    Attributes:
        model_path: Path to the ONNX model file.
        sess: ONNX runtime inference session.
        output_name: Name of the model output.
        input_name: Name of the model input.
        model_height: Model input height.
        model_width: Model input width.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the ONNX detector.

        Args:
            model_path: Path to the ONNX model file.
        """
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
        """Resize image to model input dimensions.

        Args:
            image: PIL Image or numpy array.

        Returns:
            Resized image as numpy array.
        """
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
        """Pad image to model input dimensions.

        Args:
            image: Numpy array image (H, W, C).

        Returns:
            Tuple of (padded image, padding tuple).
        """
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
        """Preprocess image for model input.

        Args:
            image: PIL Image or numpy array.

        Returns:
            Tuple of (processed image, padding).
        """
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
        """Fix bounding boxes after padding.

        Args:
            boxes: Bounding boxes array.
            width: Original image width.
            height: Original image height.
            padding: Padding tuple (left, right, top, bottom).

        Returns:
            Corrected bounding boxes.
        """
        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[2]
        boxes[..., [0, 2]] *= width / (self.model_width - padding[0] - padding[1])
        boxes[..., [1, 3]] *= height / (
            self.model_height - padding[2] - padding[3]
        )
        return boxes

    def _infer(self, image: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed image.

        Args:
            image: Preprocessed image array.

        Returns:
            Model output.
        """
        return self.sess.run([self.output_name], {self.input_name: image})[0]

    def run(self, image):
        """Run detection on image. To be implemented by subclasses."""
        raise NotImplementedError

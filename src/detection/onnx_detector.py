"""
Base ONNX detector for TorchRoyale.

Provides common functionality for loading and running ONNX models
for game object detection.
"""

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
        self.sess = ort.InferenceSession(
            self.model_path,
            providers=providers,
        )
        self.output_name = self.sess.get_outputs()[0].name

        input_ = self.sess.get_inputs()[0]
        self.input_name = input_.name
        self.model_height, self.model_width = input_.shape[2:]

    def resize(self, x: np.ndarray) -> np.ndarray:
        """Resize image to model input dimensions.

        Args:
            x: PIL Image or numpy array.

        Returns:
            Resized image as numpy array.
        """
        import PIL.Image

        if isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x)
        ratio = x.height / x.width
        if ratio > self.model_height / self.model_width:
            height = self.model_height
            width = int(self.model_height / ratio)
        else:
            width = self.model_width
            height = int(self.model_width * ratio)
        x = x.resize((width, height))
        return np.array(x)

    def pad(self, x: np.ndarray) -> tuple:
        """Pad image to model input dimensions.

        Args:
            x: Numpy array image (H, W, C).

        Returns:
            Tuple of (padded image, padding tuple).
        """
        height, width = x.shape[:2]
        dx = self.model_width - width
        dy = self.model_height - height
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        x = np.pad(
            x,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=114,
        )
        return x, padding

    def resize_pad_transpose_and_scale(self, image) -> tuple:
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
        image /= 255
        return image, padding

    def fix_bboxes(self, x: np.ndarray, width: int, height: int, padding: tuple) -> np.ndarray:
        """Fix bounding boxes after padding.

        Args:
            x: Bounding boxes array.
            width: Original image width.
            height: Original image height.
            padding: Padding tuple (left, right, top, bottom).

        Returns:
            Corrected bounding boxes.
        """
        x[:, [0, 2]] -= padding[0]
        x[:, [1, 3]] -= padding[2]
        x[..., [0, 2]] *= width / (self.model_width - padding[0] - padding[1])
        x[..., [1, 3]] *= height / (
            self.model_height - padding[2] - padding[3]
        )
        return x

    def _infer(self, x: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed image.

        Args:
            x: Preprocessed image array.

        Returns:
            Model output.
        """
        return self.sess.run([self.output_name], {self.input_name: x})[0]

    def run(self, image):
        """Run detection on image. To be implemented by subclasses."""
        raise NotImplementedError

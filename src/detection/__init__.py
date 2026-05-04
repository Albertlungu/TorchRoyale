"""Detector exports for TorchRoyale."""

from src.detection.card_detector import CardDetector
from src.detection.number_detector import NumberDetector
from src.detection.onnx_detector import OnnxDetector
from src.detection.screen_detector import ScreenDetector
from src.detection.side_detector import SideDetector
from src.detection.unit_detector import UnitDetector

__all__ = [
    "CardDetector",
    "NumberDetector",
    "OnnxDetector",
    "ScreenDetector",
    "SideDetector",
    "UnitDetector",
]

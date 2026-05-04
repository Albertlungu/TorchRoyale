"""Hand card detector backed by the Clash Royale Suite classifier."""

import json
from pathlib import Path
from typing import Sequence
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import linear_sum_assignment

from src.namespaces.cards import Card
from src.namespaces.cards import CARD_OBJECTS
from src.namespaces.cards import Cards

REPO_ROOT = Path(__file__).resolve().parents[2]
CARD_IMAGES_DIR = REPO_ROOT / "data" / "images" / "cards"
CLASSIFIER_DIR = REPO_ROOT / "data" / "models" / "card_classifier"
CLASSIFIER_MODEL_PATH = CLASSIFIER_DIR / "best_model.onnx"
CLASSIFIER_CLASSES_PATH = CLASSIFIER_DIR / "classes.json"
CARD_CONFIG = [
    (21, 609, 47, 642),
    (84, 543, 145, 616),
    (153, 543, 214, 616),
    (222, 543, 283, 616),
    (291, 543, 352, 616),
]
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOCK_CONFIDENCE = 0.90
PIXEL_CHANGE_THRESHOLD = 0.06
FINGERPRINT_GRID = 8
TORCHROYALE_SCREENSHOT_SIZE = (368, 652)

# Normalized card slot layouts from clash-royale-suite. TorchRoyale's live ADB
# path still uses CARD_CONFIG because screenshots are resized to 368x652.
CLASSIFIER_LAYOUTS = [
    (
        1080,
        1920,
        (
            (0.2074, 0.9536, 0.0565, 0.0401),
            (0.3139, 0.8724, 0.1259, 0.0984),
            (0.4398, 0.8719, 0.1185, 0.1021),
            (0.5583, 0.8724, 0.1269, 0.1016),
            (0.6861, 0.8724, 0.1213, 0.1021),
        ),
    ),
    (
        864,
        1920,
        (
            (0.0509, 0.9437, 0.0856, 0.0484),
            (0.2164, 0.8458, 0.1829, 0.1234),
            (0.3993, 0.8453, 0.1968, 0.1245),
            (0.5949, 0.8469, 0.1817, 0.1219),
            (0.7778, 0.8458, 0.1898, 0.1229),
        ),
    ),
    (
        888,
        1920,
        (
            (0.0236, 0.8750, 0.0755, 0.0401),
            (0.1273, 0.8604, 0.1363, 0.0870),
            (0.2635, 0.8604, 0.1318, 0.0885),
            (0.3941, 0.8609, 0.1306, 0.0880),
            (0.5225, 0.8609, 0.1385, 0.0870),
        ),
    ),
    (
        1080,
        2304,
        (
            (0.0444, 0.9258, 0.0991, 0.0521),
            (0.2102, 0.8242, 0.1889, 0.1285),
            (0.4000, 0.8247, 0.1907, 0.1285),
            (0.5917, 0.8251, 0.1880, 0.1285),
            (0.7796, 0.8242, 0.1870, 0.1298),
        ),
    ),
    (
        1080,
        1920,
        (
            (0.1324, 0.9271, 0.0870, 0.0620),
            (0.2769, 0.8339, 0.1537, 0.1208),
            (0.4324, 0.8339, 0.1491, 0.1250),
            (0.5806, 0.8333, 0.1546, 0.1260),
            (0.7352, 0.8339, 0.1583, 0.1250),
        ),
    ),
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def normalize_classifier_label(label: str) -> str:
    """Convert suite classifier labels to TorchRoyale card names."""
    normalized = label.strip()
    if normalized.lower().startswith("hand "):
        normalized = normalized[5:]

    lowered = normalized.lower()
    for prefix in ("evo ", "hero "):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :]
            lowered = normalized.lower()
            break

    aliases = {
        "empty": "blank",
        "mini p.e.k.k.a": "minipekka",
        "p.e.k.k.a": "pekka",
        "the log": "the_log",
        "x-bow": "x_bow",
    }
    if lowered in aliases:
        return aliases[lowered]

    return (
        lowered.replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace(" ", "_")
    )


class _SlotState:
    def __init__(self) -> None:
        self.locked = False
        self.card = Cards.BLANK
        self.confidence = 0.0
        self.fingerprint = np.array([], dtype=np.float32)


class ReferenceCardDetector:
    """Detect the current hand from card art reference images."""

    HAND_SIZE = 5
    MULTI_HASH_SCALE = 0.355
    MULTI_HASH_INTERCEPT = 163

    def __init__(
        self,
        cards: Sequence[Card],
        hash_size: int = 8,
        grey_std_threshold: float = 5,
    ) -> None:
        self.cards = list(cards) + [Cards.BLANK for _ in range(5)]
        self.hash_size = hash_size
        self.grey_std_threshold = grey_std_threshold
        self.card_hashes = self._calculate_card_hashes()

    def _calculate_multi_hash(self, image: Image.Image) -> np.ndarray:
        gray_image = self._calculate_hash(image)
        light_image = (
            self.MULTI_HASH_SCALE * gray_image + self.MULTI_HASH_INTERCEPT
        )
        dark_image = (
            gray_image - self.MULTI_HASH_INTERCEPT
        ) / self.MULTI_HASH_SCALE
        return np.vstack([gray_image, light_image, dark_image]).astype(np.float32)

    def _calculate_hash(self, image: Image.Image) -> np.ndarray:
        return np.array(
            image.resize(
                (self.hash_size, self.hash_size), Image.Resampling.BILINEAR
            ).convert("L"),
            dtype=np.float32,
        ).ravel()

    def _calculate_card_hashes(self) -> np.ndarray:
        card_hashes = np.zeros(
            (
                len(self.cards),
                3,
                self.hash_size * self.hash_size,
                self.HAND_SIZE,
            ),
            dtype=np.float32,
        )
        for index, card in enumerate(self.cards):
            path = CARD_IMAGES_DIR / f"{card.name}.jpg"
            if not path.exists():
                raise FileNotFoundError(f"Missing card reference image: {path}")
            pil_image = Image.open(path)
            multi_hash = self._calculate_multi_hash(pil_image)
            card_hashes[index] = np.tile(
                np.expand_dims(multi_hash, axis=2), (1, 1, self.HAND_SIZE)
            )
        return card_hashes

    def _detect_cards(
        self, image: Image.Image
    ) -> Tuple[list[Card], list[Image.Image]]:
        crops = [image.crop(position) for position in CARD_CONFIG]
        crop_hashes = np.array([self._calculate_hash(crop) for crop in crops]).T
        hash_diffs = np.mean(
            np.amin(np.abs(crop_hashes - self.card_hashes), axis=1), axis=1
        ).T
        _, indices = linear_sum_assignment(hash_diffs)
        return [self.cards[index] for index in indices], crops

    def _detect_if_ready(self, crops: Sequence[Image.Image]) -> list[int]:
        ready: list[int] = []
        for index, crop in enumerate(crops[1:]):
            std = np.mean(np.std(np.array(crop), axis=2))
            if std > self.grey_std_threshold:
                ready.append(index)
        return ready

    def run(self, image: Image.Image) -> Tuple[list[Card], list[int]]:
        cards, crops = self._detect_cards(image)
        return cards, self._detect_if_ready(crops)


class ClassifierCardDetector:
    """Detect the current hand with a MobileNetV3 ONNX classifier."""

    HAND_SIZE = 5

    def __init__(
        self,
        cards: Sequence[Card],
        model_path: Path = CLASSIFIER_MODEL_PATH,
        classes_path: Path = CLASSIFIER_CLASSES_PATH,
        lock_confidence: float = LOCK_CONFIDENCE,
        pixel_change_threshold: float = PIXEL_CHANGE_THRESHOLD,
        grey_std_threshold: float = 5,
    ) -> None:
        self.deck_cards = list(cards)
        self.lock_confidence = lock_confidence
        self.pixel_change_threshold = pixel_change_threshold
        self.grey_std_threshold = grey_std_threshold
        self._slots = [_SlotState() for _ in range(self.HAND_SIZE)]

        with classes_path.open("r", encoding="utf-8") as file:
            self.class_names = json.load(file)
        self.class_cards = [self._card_for_label(label) for label in self.class_names]
        deck_card_names = {card.name for card in self.deck_cards}
        deck_card_names.add(Cards.BLANK.name)
        self.allowed_class_indices = [
            index
            for index, card in enumerate(self.class_cards)
            if card.name in deck_card_names
        ]

        providers = list(
            set(ort.get_available_providers())
            & {"CUDAExecutionProvider", "CPUExecutionProvider"}
        )
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    @staticmethod
    def assets_available() -> bool:
        return CLASSIFIER_MODEL_PATH.exists() and CLASSIFIER_CLASSES_PATH.exists()

    @staticmethod
    def _select_layout(width: int, height: int):
        return min(
            CLASSIFIER_LAYOUTS,
            key=lambda layout: (layout[0] - width) ** 2 + (layout[1] - height) ** 2,
        )

    @classmethod
    def _slot_boxes(cls, image: Image.Image) -> list[tuple[int, int, int, int]]:
        if image.size == TORCHROYALE_SCREENSHOT_SIZE:
            return CARD_CONFIG

        _, _, cards = cls._select_layout(image.width, image.height)
        boxes = []
        for x, y, w, h in cards:
            left = max(0, round(x * image.width))
            top = max(0, round(y * image.height))
            right = min(image.width, round((x + w) * image.width))
            bottom = min(image.height, round((y + h) * image.height))
            boxes.append((left, top, right, bottom))
        return boxes

    @staticmethod
    def _preprocess(crop: Image.Image) -> np.ndarray:
        image = crop.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return array.transpose(2, 0, 1)

    @staticmethod
    def _fingerprint(crop: Image.Image) -> np.ndarray:
        image = crop.resize(
            (FINGERPRINT_GRID, FINGERPRINT_GRID),
            Image.Resampling.BILINEAR,
        )
        return (np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).ravel()

    @staticmethod
    def _detect_if_ready(crops: Sequence[Image.Image], threshold: float) -> list[int]:
        ready: list[int] = []
        for index, crop in enumerate(crops[1:]):
            std = np.mean(np.std(np.asarray(crop.convert("RGB")), axis=2))
            if std > threshold:
                ready.append(index)
        return ready

    @staticmethod
    def _card_for_label(label: str) -> Card:
        card_name = normalize_classifier_label(label)
        return CARD_OBJECTS.get(card_name, Cards.BLANK)

    def _prediction_for_probs(self, probabilities: np.ndarray) -> tuple[int, float]:
        candidate_indices = self.allowed_class_indices or range(len(probabilities))
        class_index = max(candidate_indices, key=lambda index: probabilities[index])
        return int(class_index), float(probabilities[class_index])

    def _unlock_changed_slots(self, crops: Sequence[Image.Image]) -> None:
        for slot, crop in zip(self._slots, crops):
            if not slot.locked:
                continue
            fingerprint = self._fingerprint(crop)
            if slot.fingerprint.size == 0:
                slot.locked = False
                continue
            distance = float(np.mean(np.abs(fingerprint - slot.fingerprint)))
            if distance > self.pixel_change_threshold:
                slot.locked = False

    def _infer_unlocked(self, crops: Sequence[Image.Image]) -> None:
        unlocked_indices = [
            index for index, slot in enumerate(self._slots) if not slot.locked
        ]
        if not unlocked_indices:
            return

        batch = np.stack(
            [self._preprocess(crops[index]) for index in unlocked_indices],
        ).astype(np.float32)
        logits = self.sess.run([self.output_name], {self.input_name: batch})[0]
        probabilities = _softmax(logits)

        for row, slot_index in enumerate(unlocked_indices):
            class_index, confidence = self._prediction_for_probs(probabilities[row])
            slot = self._slots[slot_index]
            slot.card = self.class_cards[class_index]
            slot.confidence = confidence
            if confidence >= self.lock_confidence:
                slot.locked = True
                slot.fingerprint = self._fingerprint(crops[slot_index])

    def run(self, image: Image.Image) -> Tuple[list[Card], list[int]]:
        boxes = self._slot_boxes(image)
        crops = [image.crop(box) for box in boxes]
        self._unlock_changed_slots(crops)
        self._infer_unlocked(crops)
        return [slot.card for slot in self._slots], self._detect_if_ready(
            crops,
            self.grey_std_threshold,
        )


class CardDetector:
    """Prefer the ONNX hand classifier, with hash matching as a fallback."""

    def __init__(self, cards: Sequence[Card]) -> None:
        if ClassifierCardDetector.assets_available():
            self._detector = ClassifierCardDetector(cards)
        else:
            self._detector = ReferenceCardDetector(cards)

    def run(self, image: Image.Image) -> Tuple[list[Card], list[int]]:
        return self._detector.run(image)

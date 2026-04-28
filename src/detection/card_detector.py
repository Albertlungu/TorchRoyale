"""Card detector based on reference card artwork hashes."""

from pathlib import Path
from typing import Sequence
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from src.namespaces.cards import Card
from src.namespaces.cards import Cards

REPO_ROOT = Path(__file__).resolve().parents[2]
CARD_IMAGES_DIR = REPO_ROOT / "data" / "images" / "cards"
CARD_CONFIG = [
    (21, 609, 47, 642),
    (84, 543, 145, 616),
    (153, 543, 214, 616),
    (222, 543, 283, 616),
    (291, 543, 352, 616),
]


class CardDetector:
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

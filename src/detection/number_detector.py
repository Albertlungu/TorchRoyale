"""Health and elixir number detector."""

import numpy as np
from PIL import Image
from PIL import ImageFilter

from src.namespaces.numbers import NumberDetection
from src.namespaces.numbers import Numbers

HP_WIDTH = 40
HP_HEIGHT = 10
LEFT_PRINCESS_HP_X = 74
RIGHT_PRINCESS_HP_X = 266
ALLY_PRINCESS_HP_Y = 404
ENEMY_PRINCESS_HP_Y = 95
ELIXIR_BOUNDING_BOX = (100, 628, 350, 643)
ALLY_HP_LHS_COLOUR = (111, 208, 252)
ALLY_HP_RHS_COLOUR = (63, 79, 112)
ENEMY_HP_LHS_COLOUR = (224, 35, 93)
ENEMY_HP_RHS_COLOUR = (90, 49, 68)
NUMBER_CONFIG = {
    "right_ally_princess_hp": [
        RIGHT_PRINCESS_HP_X,
        ALLY_PRINCESS_HP_Y,
        ALLY_HP_LHS_COLOUR,
        ALLY_HP_RHS_COLOUR,
    ],
    "left_ally_princess_hp": [
        LEFT_PRINCESS_HP_X,
        ALLY_PRINCESS_HP_Y,
        ALLY_HP_LHS_COLOUR,
        ALLY_HP_RHS_COLOUR,
    ],
    "right_enemy_princess_hp": [
        RIGHT_PRINCESS_HP_X,
        ENEMY_PRINCESS_HP_Y,
        ENEMY_HP_LHS_COLOUR,
        ENEMY_HP_RHS_COLOUR,
    ],
    "left_enemy_princess_hp": [
        LEFT_PRINCESS_HP_X,
        ENEMY_PRINCESS_HP_Y,
        ENEMY_HP_LHS_COLOUR,
        ENEMY_HP_RHS_COLOUR,
    ],
}


class NumberDetector:
    """Estimate tower health ratios and current elixir."""

    @staticmethod
    def _calculate_elixir(
        image: Image.Image, window_size: int = 10, threshold: float = 50
    ) -> int:
        crop = image.crop(ELIXIR_BOUNDING_BOX)
        std = np.array(crop).std(axis=(0, 2))
        rolling_std = np.convolve(
            std, np.ones(window_size) / window_size, mode="valid"
        )
        change_points = np.nonzero(rolling_std < threshold)[0]
        if len(change_points) == 0:
            return 10
        return (int(change_points[0]) + window_size) * 10 // crop.width

    @staticmethod
    def _calculate_hp(
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        lhs_colour: tuple[int, int, int],
        rhs_colour: tuple[int, int, int],
        threshold: float = 30,
    ) -> float:
        crop = np.array(
            image.crop(bbox).filter(ImageFilter.SMOOTH_MORE), dtype=np.float32
        )
        means = np.array(
            [
                np.mean(np.abs(crop - colour), axis=2)
                for colour in [lhs_colour, rhs_colour]
            ]
        )
        best_row = int(np.argmin(np.sum(np.min(means, axis=0), axis=1)))
        means = means[:, best_row, :]
        sides = np.argmin(means, axis=0)
        avg_min_dist = np.mean(np.where(sides, means[1], means[0]))

        if avg_min_dist > threshold:
            return 0.0

        change_point = int(np.argmin(np.cumsum(2 * sides - 1)))
        return change_point / (HP_WIDTH - 1)

    def run(self, image: Image.Image) -> Numbers:
        prediction: dict[str, NumberDetection] = {}
        for name, (x, y, lhs_colour, rhs_colour) in NUMBER_CONFIG.items():
            bbox = (x, y, x + HP_WIDTH, y + HP_HEIGHT)
            hp = self._calculate_hp(image, bbox, lhs_colour, rhs_colour)
            prediction[name] = NumberDetection(bbox, hp)

        elixir = self._calculate_elixir(image)
        prediction["elixir"] = NumberDetection(ELIXIR_BOUNDING_BOX, elixir)
        return Numbers(**prediction)

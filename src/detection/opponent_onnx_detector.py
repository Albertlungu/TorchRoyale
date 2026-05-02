"""Opponent detector adapter for the ONNX model from the ui branch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.detection.onnx_detector import OnnxDetector
from src.detection.side_detector import SideDetector

_UNIT_LABELS = (
    "archer",
    "archer_queen",
    "balloon",
    "bandit",
    "barbarian",
    "barbarian_hut",
    "bat",
    "battle_healer",
    "battle_ram",
    "bomb_tower",
    "bomber",
    "bowler",
    "brawler",
    "cannon",
    "cannon_cart",
    "dark_prince",
    "dart_goblin",
    "electro_dragon",
    "electro_giant",
    "electro_spirit",
    "electro_wizard",
    "elite_barbarian",
    "elixir_collector",
    "elixir_golem_large",
    "elixir_golem_medium",
    "elixir_golem_small",
    "executioner",
    "fire_spirit",
    "firecracker",
    "fisherman",
    "flying_machine",
    "furnace",
    "giant",
    "giant_skeleton",
    "giant_snowball",
    "goblin",
    "goblin_cage",
    "goblin_drill",
    "goblin_hut",
    "golden_knight",
    "golem",
    "golemite",
    "guard",
    "heal_spirit",
    "hog",
    "hog_rider",
    "baby_dragon",
    "hunter",
    "ice_golem",
    "ice_spirit",
    "ice_wizard",
    "inferno_dragon",
    "inferno_tower",
    "knight",
    "lava_hound",
    "lava_pup",
    "little_prince",
    "lumberjack",
    "magic_archer",
    "mega_knight",
    "mega_minion",
    "mighty_miner",
    "miner",
    "minion",
    "minipekka",
    "monk",
    "mortar",
    "mother_witch",
    "musketeer",
    "night_witch",
    "pekka",
    "phoenix_egg",
    "phoenix_large",
    "phoenix_small",
    "prince",
    "princess",
    "ram_rider",
    "rascal_boy",
    "rascal_girl",
    "royal_ghost",
    "royal_giant",
    "royal_guardian",
    "royal_hog",
    "royal_recruit",
    "skeleton",
    "skeleton_dragon",
    "skeleton_king",
    "sparky",
    "spear_goblin",
    "tesla",
    "tombstone",
    "valkyrie",
    "wall_breaker",
    "witch",
    "wizard",
    "x_bow",
    "zappy",
)


@dataclass(frozen=True)
class OnnxOpponentDetection:
    class_name: str
    bbox_px: tuple[int, int, int, int]
    confidence: float


class OnnxOpponentDetector(OnnxDetector):
    """Use the ui-branch ONNX detector and side classifier for enemy units."""

    MIN_CONF = 0.3
    UNIT_Y_START = 0.05
    UNIT_Y_END = 0.80

    def __init__(self, model_path: str, side_model_path: str) -> None:
        super().__init__(model_path)
        self.side_detector = SideDetector(side_model_path)

    def _preprocess(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        arena_crop = image.crop(
            (
                0,
                self.UNIT_Y_START * image.height,
                image.width,
                self.UNIT_Y_END * image.height,
            )
        )
        array, padding = self.resize_pad_transpose_and_scale(arena_crop)
        return np.expand_dims(array, axis=0), padding

    def detect(self, image_bgr: np.ndarray) -> list[OnnxOpponentDetection]:
        image_rgb = image_bgr[:, :, ::-1].copy()
        pil_image = Image.fromarray(image_rgb)
        array, padding = self._preprocess(pil_image)
        prediction = self._infer(array)[0]
        prediction = prediction[prediction[:, 4] > self.MIN_CONF]
        if len(prediction) == 0:
            return []
        prediction = self.fix_bboxes(
            prediction,
            pil_image.width,
            pil_image.height,
            padding,
        )
        prediction[:, [1, 3]] *= self.UNIT_Y_END - self.UNIT_Y_START
        prediction[:, [1, 3]] += self.UNIT_Y_START * pil_image.height

        detections: list[OnnxOpponentDetection] = []
        for left, top, right, bottom, confidence, class_id in prediction:
            bbox = (round(left), round(top), round(right), round(bottom))
            side = self.side_detector.run(pil_image.crop(bbox))
            if side != "enemy":
                continue
            if 0 <= int(class_id) < len(_UNIT_LABELS):
                class_name = _UNIT_LABELS[int(class_id)].replace("_", "-")
            else:
                class_name = f"unit-{int(class_id)}"
            detections.append(
                OnnxOpponentDetection(
                    class_name=class_name,
                    bbox_px=bbox,
                    confidence=float(confidence),
                )
            )
        return detections

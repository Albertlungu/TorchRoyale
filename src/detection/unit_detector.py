"""Unit detector backed by the migrated TorchRoyale ONNX models."""

from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from PIL import Image

from src.detection.model_loader import MODELS_DIR
from src.detection.model_loader import resolve_model_path
from src.detection.onnx_detector import OnnxDetector
from src.detection.side_detector import SideDetector
from src.namespaces.cards import Card
from src.namespaces.units import Position
from src.namespaces.units import UnitDetection
from src.namespaces.units import Units

DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280
SCREENSHOT_WIDTH = 368
SCREENSHOT_HEIGHT = 652
TILE_HEIGHT = 27.6
TILE_WIDTH = 34
TILE_INIT_X = 52
TILE_INIT_Y = 296

DETECTOR_UNITS = [
    Units.ARCHER,
    Units.ARCHER_QUEEN,
    Units.BALLOON,
    Units.BANDIT,
    Units.BARBARIAN,
    Units.BARBARIAN_HUT,
    Units.BAT,
    Units.BATTLE_HEALER,
    Units.BATTLE_RAM,
    Units.BOMB_TOWER,
    Units.BOMBER,
    Units.BOWLER,
    Units.BRAWLER,
    Units.CANNON,
    Units.CANNON_CART,
    Units.DARK_PRINCE,
    Units.DART_GOBLIN,
    Units.ELECTRO_DRAGON,
    Units.ELECTRO_GIANT,
    Units.ELECTRO_SPIRIT,
    Units.ELECTRO_WIZARD,
    Units.ELITE_BARBARIAN,
    Units.ELIXIR_COLLECTOR,
    Units.ELIXIR_GOLEM_LARGE,
    Units.ELIXIR_GOLEM_MEDIUM,
    Units.ELIXIR_GOLEM_SMALL,
    Units.EXECUTIONER,
    Units.FIRE_SPIRIT,
    Units.FIRE_CRACKER,
    Units.FISHERMAN,
    Units.FLYING_MACHINE,
    Units.FURNACE,
    Units.GIANT,
    Units.GIANT_SKELETON,
    Units.GIANT_SNOWBALL,
    Units.GOBLIN,
    Units.GOBLIN_CAGE,
    Units.GOBLIN_DRILL,
    Units.GOBLIN_HUT,
    Units.GOLDEN_KNIGHT,
    Units.GOLEM,
    Units.GOLEMITE,
    Units.GUARD,
    Units.HEAL_SPIRIT,
    Units.HOG,
    Units.HOG_RIDER,
    Units.BABY_DRAGON,
    Units.HUNTER,
    Units.ICE_GOLEM,
    Units.ICE_SPIRIT,
    Units.ICE_WIZARD,
    Units.INFERNO_DRAGON,
    Units.INFERNO_TOWER,
    Units.KNIGHT,
    Units.LAVA_HOUND,
    Units.LAVA_PUP,
    Units.LITTLE_PRINCE,
    Units.LUMBERJACK,
    Units.MAGIC_ARCHER,
    Units.MEGA_KNIGHT,
    Units.MEGA_MINION,
    Units.MIGHTY_MINER,
    Units.MINER,
    Units.MINION,
    Units.MINIPEKKA,
    Units.MONK,
    Units.MORTAR,
    Units.MOTHER_WITCH,
    Units.MUSKETEER,
    Units.NIGHT_WITCH,
    Units.PEKKA,
    Units.PHOENIX_EGG,
    Units.PHOENIX_LARGE,
    Units.PHOENIX_SMALL,
    Units.PRINCE,
    Units.PRINCESS,
    Units.RAM_RIDER,
    Units.RASCAL_BOY,
    Units.RASCAL_GIRL,
    Units.ROYAL_GHOST,
    Units.ROYAL_GIANT,
    Units.ROYAL_GUARDIAN,
    Units.ROYAL_HOG,
    Units.ROYAL_RECRUIT,
    Units.SKELETON,
    Units.SKELETON_DRAGON,
    Units.SKELETON_KING,
    Units.SPARKY,
    Units.SPEAR_GOBLIN,
    Units.TESLA,
    Units.TOMBSTONE,
    Units.VALKYRIE,
    Units.WALL_BREAKER,
    Units.WITCH,
    Units.WIZARD,
    Units.X_BOW,
    Units.ZAPPY,
]


class UnitDetector(OnnxDetector):
    """Detect and classify visible units on the arena."""

    MIN_CONF = 0.3
    UNIT_Y_START = 0.05
    UNIT_Y_END = 0.80

    def __init__(
        self,
        cards: Sequence[Card],
        model_path: Optional[Path] = None,
        side_model_path: Optional[Path] = None,
    ) -> None:
        super().__init__(str(model_path or resolve_model_path("units_M_480x352.onnx")))
        self.cards = list(cards)
        self.side_detector = SideDetector(
            str(side_model_path or resolve_model_path("side.onnx"))
        )
        self.possible_ally_names = self._get_possible_ally_names()

    @staticmethod
    def _get_tile_xy(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x = (bbox[0] + bbox[2]) * DISPLAY_WIDTH / (2 * SCREENSHOT_WIDTH)
        y = bbox[3] * DISPLAY_HEIGHT / SCREENSHOT_HEIGHT
        tile_x = round(((x - TILE_INIT_X) / TILE_WIDTH) - 0.5)
        tile_y = round(((DISPLAY_HEIGHT - TILE_INIT_Y - y) / TILE_HEIGHT) - 0.5)
        return tile_x, tile_y

    def _get_possible_ally_names(self) -> set[str]:
        possible_ally_names: set[str] = set()
        for card in self.cards:
            for unit in card.units:
                possible_ally_names.add(unit.name)
        return possible_ally_names

    def _calculate_side(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        name: str,
    ) -> str:
        if name not in self.possible_ally_names:
            return "enemy"
        crop = image.crop(bbox)
        return self.side_detector.run(crop)

    def _preprocess(self, image: Image.Image) -> Tuple[np.ndarray, tuple[int, int, int, int]]:
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

    def _post_process(
        self,
        prediction: np.ndarray,
        image_height: int,
        image: Image.Image,
    ) -> Tuple[List[UnitDetection], List[UnitDetection]]:
        prediction[:, [1, 3]] *= self.UNIT_Y_END - self.UNIT_Y_START
        prediction[:, [1, 3]] += self.UNIT_Y_START * image_height

        allies: List[UnitDetection] = []
        enemies: List[UnitDetection] = []

        for left, top, right, bottom, confidence, class_id in prediction:
            bbox = (round(left), round(top), round(right), round(bottom))
            tile_x, tile_y = self._get_tile_xy(bbox)
            position = Position(bbox, float(confidence), tile_x, tile_y)
            unit = DETECTOR_UNITS[int(class_id)]
            detection = UnitDetection(unit, position)
            side = self._calculate_side(image, bbox, unit.name)
            if side == "ally":
                allies.append(detection)
            else:
                enemies.append(detection)

        return allies, enemies

    def run(self, image: Image.Image) -> Tuple[List[UnitDetection], List[UnitDetection]]:
        height, width = image.height, image.width
        array, padding = self._preprocess(image)
        prediction = self._infer(array)[0]
        prediction = prediction[prediction[:, 4] > self.MIN_CONF]
        prediction = self.fix_bboxes(prediction, width, height, padding)
        return self._post_process(prediction, height, image)

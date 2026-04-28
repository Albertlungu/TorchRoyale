"""Base action protocol for heuristic live bot play."""

from abc import ABC
from abc import abstractmethod

from src.namespaces.cards import Card


class Action(ABC):
    CARD: Card = None

    def __init__(self, index: int, tile_x: int, tile_y: int) -> None:
        self.index = index
        self.tile_x = tile_x
        self.tile_y = tile_y

    def __repr__(self) -> str:
        return f"{self.CARD.name} at ({self.tile_x}, {self.tile_y})"

    @abstractmethod
    def calculate_score(self, state):
        raise NotImplementedError

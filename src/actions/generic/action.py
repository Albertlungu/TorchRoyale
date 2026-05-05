"""Base action protocol for heuristic live bot play."""

from abc import ABC
from abc import abstractmethod

from src.namespaces.cards import Card


class Action(ABC):
    """
    Abstract base class for all card placement actions.

    Attributes:
        CARD (Card): The card associated with this action.
        index (int): Hand index of the card.
        tile_x (int): Grid column for the placement.
        tile_y (int): Grid row for the placement.
    """

    CARD: Card = None

    def __init__(self, index: int, tile_x: int, tile_y: int) -> None:
        """
        Initialise an action at a specific grid tile.

        Args:
            index (int): Hand index of the card.
            tile_x (int): Grid column for the placement.
            tile_y (int): Grid row for the placement.
        """
        self.index = index
        self.tile_x = tile_x
        self.tile_y = tile_y

    def __repr__(self) -> str:
        """
        Return a human-readable string representation of the action.

        Returns:
            str: Card name and tile coordinates.
        """
        return f"{self.CARD.name} at ({self.tile_x}, {self.tile_y})"

    @abstractmethod
    def calculate_score(self, state) -> list:
        """
        Compute a priority score for this action given the current game state.

        Args:
            state: Current game state containing unit detections and resource counts.

        Returns:
            list: Ordered score components used for action ranking.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

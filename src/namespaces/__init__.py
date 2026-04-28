"""TorchRoyale data namespaces."""

from src.namespaces.cards import Card
from src.namespaces.cards import CARD_OBJECTS
from src.namespaces.cards import Cards
from src.namespaces.cards import NAME2CARD
from src.namespaces.numbers import NumberDetection
from src.namespaces.numbers import Numbers
from src.namespaces.screens import Screen
from src.namespaces.screens import Screens
from src.namespaces.state import State
from src.namespaces.units import NAME2UNIT
from src.namespaces.units import Position
from src.namespaces.units import Target
from src.namespaces.units import Transport
from src.namespaces.units import Unit
from src.namespaces.units import UnitCategory
from src.namespaces.units import UnitDetection
from src.namespaces.units import UNIT_OBJECTS
from src.namespaces.units import Units

__all__ = [
    "CARD_OBJECTS",
    "Card",
    "Cards",
    "NAME2CARD",
    "NumberDetection",
    "Numbers",
    "NAME2UNIT",
    "Position",
    "Screen",
    "Screens",
    "State",
    "Target",
    "Transport",
    "Unit",
    "UnitCategory",
    "UnitDetection",
    "UNIT_OBJECTS",
    "Units",
]

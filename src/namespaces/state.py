"""Aggregated game state dataclass for TorchRoyale."""

from dataclasses import dataclass
from typing import List
from typing import Tuple

from src.namespaces.cards import Card
from src.namespaces.numbers import Numbers
from src.namespaces.screens import Screen
from src.namespaces.units import UnitDetection


@dataclass
class State:
    allies: List[UnitDetection]
    enemies: List[UnitDetection]
    numbers: Numbers
    cards: Tuple[Card, ...]
    ready: List[int]
    screen: Screen

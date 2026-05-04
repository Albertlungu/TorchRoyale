"""
Game phase derivation from the timer sequence.

Determines whether each frame is in single-elixir, double-elixir, triple-elixir,
or game-over phase. Overtime is inferred from a timer jump from near-zero back
to a large value.

Public API:
  Phase          -- string enum of the four possible game phases
  derive_phases() -- compute multiplier and phase lists from a timer sequence
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from src.constants.game import DOUBLE_ELIXIR_START_S, TRIPLE_ELIXIR_START_S


class Phase(str, Enum):
    """The four mutually exclusive game phases."""

    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    GAME_OVER = "game_over"


def derive_phases(
    timer_filled: List[Optional[int]],
) -> Tuple[List[int], List[str]]:
    """
    Derive elixir_multiplier and game_phase for every frame from the
    filled timer sequence.

    Overtime is detected by the timer jumping from near-zero back to
    a value >= 100 seconds.

    Args:
        timer_filled: per-frame timer values in seconds. None entries are
                      forward-filled from the previous known value.

    Returns:
        multipliers: list of int (1, 2, or 3), one per frame.
        phases:      list of Phase string values, one per frame.
    """
    in_overtime: bool = False
    prev: Optional[int] = None
    multipliers: List[int] = []
    phases: List[str] = []

    for secs in timer_filled:
        if prev is not None and secs is not None and prev <= 10 and secs >= 100:
            in_overtime = True

        if secs is None:
            mult: int = multipliers[-1] if multipliers else 1
            phase: str = phases[-1] if phases else Phase.SINGLE.value
        elif secs <= 0:
            mult, phase = 1, Phase.GAME_OVER.value
        elif in_overtime:
            if secs <= TRIPLE_ELIXIR_START_S:
                mult, phase = 3, Phase.TRIPLE.value
            else:
                mult, phase = 2, Phase.DOUBLE.value
        else:
            if secs <= DOUBLE_ELIXIR_START_S:
                mult, phase = 2, Phase.DOUBLE.value
            else:
                mult, phase = 1, Phase.SINGLE.value

        multipliers.append(mult)
        phases.append(phase)
        if secs is not None:
            prev = secs

    return multipliers, phases

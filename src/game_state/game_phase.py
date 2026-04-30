"""
Game phase state machine for tracking Clash Royale match phases.

Tracks transitions between:
- Single Elixir (game start, 3:00 - 2:00)
- Double Elixir (2:00 - 0:00, or sudden death)
- Triple Elixir (last 1:00 of sudden death)
- Sudden Death (overtime when tied)
- Game Over
"""

from typing import Optional
from ..constants.game_constants import GamePhase, GameTimingConstants


class GamePhaseTracker:
    """
    State machine for tracking game phases.

    Uses multiplier icon detection as primary signal, with timer
    as fallback for more precise timing.

    Phase Transitions:
    - SINGLE_ELIXIR -> DOUBLE_ELIXIR: Timer reaches 1:00 or x2 icon appears
    - DOUBLE_ELIXIR -> GAME_OVER: Timer reaches 0:00 (if not tied)
    - DOUBLE_ELIXIR -> SUDDEN_DEATH: Timer reaches 0:00 (if tied)
    - SUDDEN_DEATH -> TRIPLE_ELIXIR: Last 1:00 of overtime or x3 icon appears
    - TRIPLE_ELIXIR -> GAME_OVER: Overtime ends or tower destroyed

    Attributes:
        current_phase (GamePhase): Current game phase.
        is_sudden_death (bool): Whether in sudden death overtime.
        overtime_start_ms (Optional[int]): Timestamp when overtime started.
        _last_timer_seconds (Optional[int]): Last detected timer value.
    """

    def __init__(self):
        """Initialize the game phase tracker."""
        self.current_phase = GamePhase.SINGLE_ELIXIR
        self.is_sudden_death = False
        self.overtime_start_ms: Optional[int] = None
        self._last_timer_seconds: Optional[int] = None

    def reset(self):
        """Reset tracker for a new game."""
        self.current_phase = GamePhase.SINGLE_ELIXIR
        self.is_sudden_death = False
        self.overtime_start_ms = None
        self._last_timer_seconds = None

    def update(
        self,
        multiplier_detected: int,
        timer_seconds: Optional[int] = None,
        towers_tied: bool = False,
        timestamp_ms: int = 0
    ) -> GamePhase:
        """
        Update game phase based on detected signals.

        Args:
            multiplier_detected: 1, 2, or 3 from icon detection
            timer_seconds: Seconds remaining (None if not detected)
            towers_tied: Whether tower count is tied (for sudden death detection)
            timestamp_ms: Current timestamp (for overtime tracking)

        Returns:
            Current GamePhase after update
        """
        # Update timer tracking
        if timer_seconds is not None:
            self._last_timer_seconds = timer_seconds

        # Primary signal: timer (more reliable than multiplier icon detection)
        if timer_seconds is not None:
            # Check for phase transitions based on timer
            if timer_seconds <= 0:
                if self.is_sudden_death:
                    # End of sudden death
                    self.current_phase = GamePhase.GAME_OVER
                elif towers_tied:
                    # Transition to sudden death
                    self.is_sudden_death = True
                    self.overtime_start_ms = timestamp_ms
                    self.current_phase = GamePhase.SUDDEN_DEATH
                else:
                    # Game over (someone won in regulation)
                    self.current_phase = GamePhase.GAME_OVER
            elif timer_seconds <= GameTimingConstants.DOUBLE_ELIXIR_START:
                # Double elixir phase (last 60 seconds)
                if not self.is_sudden_death:
                    self.current_phase = GamePhase.DOUBLE_ELIXIR
            else:
                # Single elixir phase (more than 60 seconds remaining)
                if not self.is_sudden_death:
                    self.current_phase = GamePhase.SINGLE_ELIXIR

        # Secondary signal: multiplier icon (only trust if no timer or confirms timer-based phase)
        # This helps catch triple elixir and sudden death transitions
        if multiplier_detected == 3:
            self.current_phase = GamePhase.TRIPLE_ELIXIR
            self.is_sudden_death = True
        elif multiplier_detected == 2 and self.is_sudden_death:
            # x2 icon in sudden death (before triple elixir phase)
            self.current_phase = GamePhase.SUDDEN_DEATH
        elif multiplier_detected == 1 and timer_seconds is None:
            # Only trust x1 icon if no timer available
            if not self.is_sudden_death:
                self.current_phase = GamePhase.SINGLE_ELIXIR

        # Sudden death overtime tracking
        if self.is_sudden_death and self.overtime_start_ms is not None:
            overtime_elapsed = timestamp_ms - self.overtime_start_ms
            overtime_remaining_ms = (
                GameTimingConstants.SUDDEN_DEATH_DURATION * 1000 - overtime_elapsed
            )

            # Triple elixir in last minute of sudden death
            if overtime_remaining_ms <= GameTimingConstants.TRIPLE_ELIXIR_START * 1000:
                self.current_phase = GamePhase.TRIPLE_ELIXIR
            elif overtime_remaining_ms <= 0:
                self.current_phase = GamePhase.GAME_OVER

        return self.current_phase

    def force_sudden_death(self, timestamp_ms: int):
        """
        Force transition to sudden death (e.g., when tie is detected).

        Args:
            timestamp_ms: Current timestamp for overtime tracking
        """
        self.is_sudden_death = True
        self.overtime_start_ms = timestamp_ms
        self.current_phase = GamePhase.SUDDEN_DEATH

    @property
    def elixir_multiplier(self) -> int:
        """
        Get current elixir multiplier.

        Returns:
            1, 2, or 3 based on current phase
        """
        if self.current_phase == GamePhase.TRIPLE_ELIXIR:
            return 3
        elif self.current_phase in (GamePhase.DOUBLE_ELIXIR, GamePhase.SUDDEN_DEATH):
            return 2
        else:
            return 1

    @property
    def is_game_over(self) -> bool:
        """Check if game has ended."""
        return self.current_phase == GamePhase.GAME_OVER

    @property
    def is_overtime(self) -> bool:
        """Check if in sudden death overtime."""
        return self.is_sudden_death

    def __repr__(self) -> str:
        return (
            f"GamePhaseTracker(phase={self.current_phase.value}, "
            f"multiplier={self.elixir_multiplier}, "
            f"overtime={self.is_sudden_death})"
        )

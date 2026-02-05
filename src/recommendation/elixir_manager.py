"""
Elixir management and tracking for both player and opponent.

The opponent's elixir is calculated based on:
- Starting elixir (5)
- Time-based regeneration (varies by game phase)
- Card plays (subtract elixir cost when detected)
"""

from typing import List, Optional, Set
from dataclasses import dataclass, field
from ..constants.game_constants import (
    GamePhase,
    ElixirConstants,
    get_elixir_cost,
    get_regen_rate,
)


@dataclass
class CardPlayEvent:
    """Records when a card is played."""
    timestamp_ms: int
    card_name: str
    elixir_cost: int
    is_opponent: bool


@dataclass
class ElixirState:
    """Current elixir state for a player."""
    current: float
    last_update_ms: int
    cards_played: List[CardPlayEvent] = field(default_factory=list)


class OpponentElixirTracker:
    """
    Tracks opponent's elixir through calculation.

    Since we can't see the opponent's elixir directly, we calculate it:
    1. Start at 5 elixir (game start)
    2. Add elixir based on regeneration rate and time elapsed
    3. Subtract elixir when opponent plays cards (detected on field)

    Key assumptions:
    - Opponent starts with 5 elixir
    - Max elixir is 10 (can't exceed this)
    - Each card costs a known amount of elixir
    """

    def __init__(self):
        """Initialize the opponent elixir tracker."""
        self.current_elixir: float = ElixirConstants.STARTING_ELIXIR
        self.last_update_ms: int = 0
        self.game_phase: GamePhase = GamePhase.SINGLE_ELIXIR
        self.card_history: List[CardPlayEvent] = []

        # Track cards currently on field to detect new plays
        self._previous_on_field_cards: Set[str] = set()

        # Track unique card IDs to avoid double-counting
        # Format: "card_name_tile_x_tile_y"
        self._tracked_card_instances: Set[str] = set()

    def reset(self):
        """Reset tracker for a new game."""
        self.current_elixir = ElixirConstants.STARTING_ELIXIR
        self.last_update_ms = 0
        self.game_phase = GamePhase.SINGLE_ELIXIR
        self.card_history = []
        self._previous_on_field_cards = set()
        self._tracked_card_instances = set()

    def update(
        self,
        timestamp_ms: int,
        game_phase: GamePhase,
        opponent_detections: List,
    ) -> float:
        """
        Update opponent elixir estimate based on current state.

        Args:
            timestamp_ms: Current frame timestamp in milliseconds
            game_phase: Current game phase (affects regen rate)
            opponent_detections: List of opponent card detections from this frame
                                Each detection should have: class_name, is_on_field,
                                tile_x, tile_y attributes

        Returns:
            Estimated opponent elixir (0.0 - 10.0)
        """
        # Calculate time delta
        delta_ms = timestamp_ms - self.last_update_ms
        if delta_ms < 0:
            # Time went backwards (shouldn't happen), reset
            delta_ms = 0

        delta_seconds = delta_ms / 1000.0

        # Get regeneration rate based on game phase
        regen_rate = get_regen_rate(game_phase)

        # Add regenerated elixir
        if delta_seconds > 0 and regen_rate > 0:
            elixir_gained = delta_seconds / regen_rate
            self.current_elixir = min(
                ElixirConstants.MAX_ELIXIR,
                self.current_elixir + elixir_gained
            )

        # Detect newly played cards
        newly_played_cards = self._detect_new_cards(opponent_detections)

        # Subtract elixir for played cards
        for card_name in newly_played_cards:
            cost = get_elixir_cost(card_name)
            if cost > 0:
                # Don't go below 0 (might indicate tracking error)
                self.current_elixir = max(0, self.current_elixir - cost)

                # Record the play
                self.card_history.append(CardPlayEvent(
                    timestamp_ms=timestamp_ms,
                    card_name=card_name,
                    elixir_cost=cost,
                    is_opponent=True
                ))

        # Update state
        self.last_update_ms = timestamp_ms
        self.game_phase = game_phase

        return self.current_elixir

    def _detect_new_cards(self, detections: List) -> List[str]:
        """
        Detect cards that were just played (newly appeared on field).

        Uses position-based tracking to avoid double-counting the same
        card that persists across frames.

        Args:
            detections: List of opponent detections

        Returns:
            List of newly played card names
        """
        newly_played = []
        current_card_instances: Set[str] = set()

        for det in detections:
            # Only consider on-field cards from opponent
            if not getattr(det, 'is_opponent', False):
                continue
            if not getattr(det, 'is_on_field', False):
                continue

            card_name = getattr(det, 'class_name', '')
            tile_x = getattr(det, 'tile_x', 0)
            tile_y = getattr(det, 'tile_y', 0)

            # Create unique identifier for this card instance
            # Using tile position helps track the same card across frames
            card_id = f"{card_name}_{tile_x}_{tile_y}"
            current_card_instances.add(card_id)

            # Check if this is a new card we haven't seen
            if card_id not in self._tracked_card_instances:
                newly_played.append(card_name)

        # Update tracked instances
        # Keep cards that are still visible + add new ones
        self._tracked_card_instances = current_card_instances

        return newly_played

    def get_elixir_spent(self, since_ms: Optional[int] = None) -> int:
        """
        Get total elixir spent by opponent.

        Args:
            since_ms: Only count plays after this timestamp (None = all)

        Returns:
            Total elixir spent
        """
        total = 0
        for event in self.card_history:
            if since_ms is None or event.timestamp_ms >= since_ms:
                total += event.elixir_cost
        return total

    def get_recent_plays(self, last_n: int = 5) -> List[CardPlayEvent]:
        """
        Get the most recent card plays.

        Args:
            last_n: Number of recent plays to return

        Returns:
            List of recent CardPlayEvents (most recent first)
        """
        return list(reversed(self.card_history[-last_n:]))

    def estimate_elixir_at_time(
        self,
        target_ms: int,
        from_ms: Optional[int] = None
    ) -> float:
        """
        Estimate what opponent's elixir was at a specific time.

        This is a rough estimate based on card plays in history.

        Args:
            target_ms: Target timestamp
            from_ms: Start timestamp (None = game start)

        Returns:
            Estimated elixir at that time
        """
        if from_ms is None:
            from_ms = 0

        elixir = ElixirConstants.STARTING_ELIXIR

        # Simple linear regen estimate (assumes single elixir for simplicity)
        elapsed_seconds = (target_ms - from_ms) / 1000.0
        elixir_gained = elapsed_seconds / ElixirConstants.SINGLE_REGEN_RATE
        elixir = min(ElixirConstants.MAX_ELIXIR, elixir + elixir_gained)

        # Subtract cards played up to that time
        for event in self.card_history:
            if from_ms <= event.timestamp_ms <= target_ms:
                elixir = max(0, elixir - event.elixir_cost)
                # Re-add regen after card play (simplified)

        return elixir

    @property
    def total_cards_played(self) -> int:
        """Get total number of cards played by opponent."""
        return len(self.card_history)

    @property
    def average_elixir_per_card(self) -> float:
        """Get average elixir cost of opponent's cards."""
        if not self.card_history:
            return 0.0
        total_cost = sum(e.elixir_cost for e in self.card_history)
        return total_cost / len(self.card_history)

    def __repr__(self) -> str:
        return (
            f"OpponentElixirTracker("
            f"elixir={self.current_elixir:.1f}, "
            f"phase={self.game_phase.value}, "
            f"cards_played={len(self.card_history)})"
        )


class PlayerElixirTracker:
    """
    Tracks player's elixir from visual detection.

    Unlike opponent tracking, player elixir can be read directly
    from the UI, so this class mainly validates and smooths the readings.
    """

    def __init__(self):
        """Initialize player elixir tracker."""
        self.current_elixir: int = ElixirConstants.STARTING_ELIXIR
        self.last_detected: int = ElixirConstants.STARTING_ELIXIR
        self.detection_history: List[int] = []
        self.confidence: float = 1.0

    def reset(self):
        """Reset for new game."""
        self.current_elixir = ElixirConstants.STARTING_ELIXIR
        self.last_detected = ElixirConstants.STARTING_ELIXIR
        self.detection_history = []
        self.confidence = 1.0

    def update(self, detected_elixir: int, detection_confidence: float) -> int:
        """
        Update player elixir from visual detection.

        Applies smoothing to handle occasional misreads.

        Args:
            detected_elixir: Elixir value detected from UI (-1 if failed)
            detection_confidence: Confidence of the detection (0-1)

        Returns:
            Current elixir value (smoothed)
        """
        if detected_elixir < 0:
            # Detection failed, use last known value
            return self.current_elixir

        # Validate range
        detected_elixir = max(0, min(ElixirConstants.MAX_ELIXIR, detected_elixir))

        # Add to history
        self.detection_history.append(detected_elixir)
        if len(self.detection_history) > 5:
            self.detection_history.pop(0)

        # Use median of recent detections for smoothing
        if len(self.detection_history) >= 3:
            self.current_elixir = int(sorted(self.detection_history)[len(self.detection_history) // 2])
        else:
            self.current_elixir = detected_elixir

        self.last_detected = detected_elixir
        self.confidence = detection_confidence

        return self.current_elixir

    def __repr__(self) -> str:
        return f"PlayerElixirTracker(elixir={self.current_elixir}, confidence={self.confidence:.2f})"

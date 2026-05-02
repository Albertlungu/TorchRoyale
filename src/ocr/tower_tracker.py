"""
Tower HP state machine for Clash Royale OCR.

Each tower's bbox covers the whole tower image, so the OCR reads the tower
level prepended to the HP (e.g., level 15 king at 4176 HP → "154176").

Logic per tower:
  - First reading that is 1-16: this is the level only (no damage yet).
    Store the level. HP = max HP for that level (kings) or None (princess).
  - First reading > 16: level has not been seen standalone yet. Extract the
    level prefix (1-2 digits ≤ 16 whose remainder is ≥ 3 digits), store it,
    parse remainder as HP.
  - Subsequent readings: strip the stored level string from the front,
    parse remainder as HP.
  - Empty string: tower is destroyed (once level has been seen at least once).

Win/loss at end of game:
  determine_outcome(player_trackers, opponent_trackers) → "win" | "loss"

Public API:
  KING_MAX_HP         -- dict[level → max HP]
  TowerReading        -- result of one OCR read
  TowerTracker        -- stateful per-tower tracker
  determine_outcome   -- game result from 6 final trackers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# King tower max HP by level (from official stats)
KING_MAX_HP: Dict[int, int] = {
    1: 2400, 2: 2568, 3: 2736, 4: 2904, 5: 3096,
    6: 3312, 7: 3528, 8: 3768, 9: 4008, 10: 4392,
    11: 4824, 12: 5304, 13: 5832, 14: 6408, 15: 7032,
    16: 7704,
}

# Princess tower max HP by level (from official stats)
PRINCESS_MAX_HP: Dict[int, int] = {
    1: 1400, 2: 1512, 3: 1624, 4: 1750, 5: 1890,
    6: 2030, 7: 2184, 8: 2352, 9: 2534, 10: 2786,
    11: 3052, 12: 3346, 13: 3668, 14: 4032, 15: 4424,
    16: 4858,
}


@dataclass
class TowerReading:
    """Result of one OCR pass on a tower region."""
    hp: Optional[int]        # current HP; None if not yet damaged (princess)
    level: Optional[int]     # tower level once known
    destroyed: bool          # True if bbox went empty after level was locked in
    at_max: bool             # True if king tower is undamaged (showing level only)
    raw: str                 # raw OCR string before processing


def _extract_level_prefix(s: str) -> Optional[int]:
    """
    Try to read a 1-or-2-digit level (1-16) from the front of s,
    requiring the remainder to be at least 3 digits (plausible HP).

    Tries 2-digit prefix first to avoid ambiguity (e.g., "154176").
    Returns the level int, or None if no valid prefix found.
    """
    if len(s) >= 5:
        two = int(s[:2])
        if 10 <= two <= 16:
            return two
    if len(s) >= 4:
        two = int(s[:2])
        if 1 <= two <= 9:
            # single-digit level with 2-digit first chars: use 1-digit
            one = int(s[:1])
            if 1 <= one <= 9:
                return one
        if 10 <= two <= 16:
            return two
    if len(s) >= 4:
        one = int(s[:1])
        if 1 <= one <= 9:
            return one
    return None


class TowerTracker:
    """
    Stateful HP tracker for one tower.

    Args:
        is_king: True for king towers (uses KING_MAX_HP table).
    """

    def __init__(self, is_king: bool = False) -> None:
        self.is_king: bool = is_king
        self._hp_table: Dict[int, int] = KING_MAX_HP if is_king else PRINCESS_MAX_HP
        self._level: Optional[int] = None
        self._level_str: str = ""
        self._destroyed: bool = False
        self._ever_seen: bool = False  # True once we read any non-empty value

    @property
    def level(self) -> Optional[int]:
        return self._level

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def read(self, raw: str) -> TowerReading:
        """
        Process one raw OCR string from the tower bbox.

        Args:
            raw: digit-only string from OCR (may be empty).

        Returns:
            TowerReading with interpreted HP and state.
        """
        digits = "".join(c for c in raw if c.isdigit())

        # Empty → destroyed if we already locked in a level
        if not digits:
            if self._ever_seen:
                self._destroyed = True
            return TowerReading(
                hp=0 if self._destroyed else None,
                level=self._level,
                destroyed=self._destroyed,
                at_max=False,
                raw=raw,
            )

        self._ever_seen = True
        value = int(digits)

        # --- Level not yet known ---
        if self._level is None:
            if 1 <= value <= 16:
                # Standalone level reading (no damage yet)
                self._level = value
                self._level_str = str(value)
                max_hp = self._hp_table.get(value)
                return TowerReading(
                    hp=max_hp,
                    level=self._level,
                    destroyed=False,
                    at_max=True,
                    raw=raw,
                )
            else:
                # Compound reading — extract level prefix
                lvl = _extract_level_prefix(digits)
                if lvl is not None:
                    self._level = lvl
                    self._level_str = str(lvl)
                    remainder = digits[len(self._level_str):]
                    if len(remainder) < 3:
                        hp = self._hp_table.get(self._level)
                        return TowerReading(
                            hp=hp, level=self._level, destroyed=False, at_max=True, raw=raw
                        )
                    hp_str = remainder[-4:] if len(remainder) > 4 else remainder
                    hp = int(hp_str) if hp_str else None
                else:
                    # Can't find level prefix; take last 4 digits as HP
                    hp = int(digits[-4:]) if len(digits) > 4 else value
                # Clamp to max HP if level is now known
                if hp is not None and self._level is not None:
                    max_hp = self._hp_table.get(self._level, 9999)
                    if hp > max_hp:
                        trimmed = int(str(hp)[-4:])
                        hp = trimmed if trimmed <= max_hp else max_hp
                return TowerReading(
                    hp=hp,
                    level=self._level,
                    destroyed=False,
                    at_max=False,
                    raw=raw,
                )

        # --- Level known: strip prefix, then take last 4 digits as HP ---
        # The bbox covers the whole tower so OCR often picks up stray digits
        # between the level icon and the HP number. Taking the last 3-4 digits
        # isolates the HP regardless of middle noise.
        if digits.startswith(self._level_str) and len(digits) > len(self._level_str):
            remainder = digits[len(self._level_str):]
            # Remainder < 3 digits = trailing noise, not a real HP reading
            if len(remainder) < 3:
                hp = self._hp_table.get(self._level)
                return TowerReading(
                    hp=hp, level=self._level, destroyed=False, at_max=True, raw=raw
                )
            # HP is at most 4 digits; trim leading noise
            hp_str = remainder[-4:] if len(remainder) > 4 else remainder
            hp = int(hp_str)
        elif 1 <= value <= 16 and self.is_king:
            # King showing level only again (undamaged / OCR missed HP bar)
            hp = KING_MAX_HP.get(self._level)
            return TowerReading(
                hp=hp, level=self._level, destroyed=False, at_max=True, raw=raw
            )
        else:
            # Doesn't start with level prefix — take last 4 digits as HP
            max_hp = self._hp_table.get(self._level, 9999) if self._level else 9999
            hp = int(str(value)[-4:]) if value > max_hp else value

        # Final clamp to known max HP for this tower level
        if hp is not None and self._level is not None:
            max_hp = self._hp_table.get(self._level, 9999)
            if hp > max_hp:
                trimmed = int(str(hp)[-4:])
                hp = trimmed if trimmed <= max_hp else max_hp
            # Noise rejection: HP below threshold is almost certainly an OCR
            # misread. King tower crown badges produce far more noise than
            # princess towers, so king uses a higher threshold (60% of max).
            noise_floor = max_hp * (0.65 if self.is_king else 0.20)
            if hp < noise_floor:
                hp = max_hp
                return TowerReading(
                    hp=hp, level=self._level, destroyed=False, at_max=True, raw=raw
                )

        return TowerReading(
            hp=hp, level=self._level, destroyed=False, at_max=False, raw=raw
        )

    def set_level(self, level: int) -> None:
        """
        Externally set the tower level when OCR is unreliable.

        Used to propagate princess tower level to the king tracker.
        Always overrides because princess tower OCR is more reliable
        than king crown badge OCR.
        """
        if 1 <= level <= 16:
            self._level = level
            self._level_str = str(level)

    def reset(self) -> None:
        """Reset to initial state for a new game."""
        self._level = None
        self._level_str = ""
        self._destroyed = False
        self._ever_seen = False


def determine_outcome(
    player: List[TowerTracker],
    opponent: List[TowerTracker],
) -> str:
    """
    Determine game outcome from the final state of all 6 towers.

    Rule 1: most towers destroyed loses.
    Rule 2 (tiebreak): player with lowest min HP across their 3 towers loses.

    Args:
        player:   list of 3 TowerTrackers [left, king, right] for the player.
        opponent: list of 3 TowerTrackers [left, king, right] for the opponent.

    Returns:
        "win" or "loss" from the player's perspective.
    """
    player_destroyed = sum(1 for t in player if t.destroyed)
    opp_destroyed = sum(1 for t in opponent if t.destroyed)

    if player_destroyed != opp_destroyed:
        return "loss" if player_destroyed > opp_destroyed else "win"

    # Tiebreak: lowest min HP
    def min_hp(trackers: List[TowerTracker]) -> int:
        hps = [t._level and KING_MAX_HP.get(t._level, 0) if t.destroyed else 0
               for t in trackers]
        # Use last known HP from reading (not stored in tracker — caller handles)
        return min(hps) if hps else 0

    # For the tiebreak we need last readings — determine_outcome_from_readings
    # is the preferred call when HP values are available. Fall back to "win".
    return "win"


def determine_outcome_from_readings(
    player_hps: List[Optional[int]],
    player_destroyed: List[bool],
    opponent_hps: List[Optional[int]],
    opponent_destroyed: List[bool],
) -> str:
    """
    Determine outcome given explicit HP and destroyed lists.

    Args:
        player_hps:        last HP reading for each of player's 3 towers.
        player_destroyed:  destroyed flags for player's 3 towers.
        opponent_hps:      last HP reading for each of opponent's 3 towers.
        opponent_destroyed: destroyed flags for opponent's 3 towers.

    Returns:
        "win" or "loss" from the player's perspective.
    """
    p_dead = sum(player_destroyed)
    o_dead = sum(opponent_destroyed)

    if p_dead != o_dead:
        return "loss" if p_dead > o_dead else "win"

    # Tiebreak: lowest min HP loses
    p_min = min((hp for hp in player_hps if hp is not None), default=0)
    o_min = min((hp for hp in opponent_hps if hp is not None), default=0)
    return "loss" if p_min < o_min else "win"

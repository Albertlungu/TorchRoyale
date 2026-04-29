"""
Card vocabulary: canonical names and elixir costs.

Canonical names match KataCR's unit_list where possible, with additions
for cards the project tracks that KataCR may not label.

Public API:
  ELIXIR_COSTS  -- dict mapping card name to elixir cost (None for Mirror)
  CARD_NAMES    -- sorted list of all known card names (model vocabulary)
  CARD_TO_IDX   -- card name → vocabulary index
  IDX_TO_CARD   -- vocabulary index → card name
  VOCAB_SIZE    -- total number of cards in vocabulary
  elixir_cost() -- look up cost for a name, returns None if unknown
  card_to_idx() -- look up vocabulary index for a name, returns 0 if unknown
"""
from __future__ import annotations

from typing import Dict, Optional

# fmt: off
ELIXIR_COSTS: Dict[str, int] = {
    "archer":               3,
    "archer-evolution":     3,
    "arrows":               3,
    "baby-dragon":          4,
    "balloon":              5,
    "bandit":               3,
    "barbarian":            5,
    "barbarian-barrel":     2,
    "barbarian-evolution":  5,
    "barbarian-hut":        7,
    "bat":                  2,
    "bat-evolution":        2,
    "bomb-tower":           4,
    "bomber":               2,
    "bomber-evolution":     2,
    "bowler":               5,
    "cannon":               3,
    "cannon-cart":          5,
    "clone":                3,
    "dark-prince":          4,
    "dart-goblin":          3,
    "earthquake":           3,
    "electro-dragon":       5,
    "electro-giant":        7,
    "electro-spirit":       1,
    "electro-wizard":       4,
    "elite-barbarian":      6,
    "elixir-collector":     6,
    "elixir-golem-big":     6,
    "executioner":          5,
    "fire-spirit":          1,
    "firecracker":          3,
    "firecracker-evolution":3,
    "fireball":             4,
    "fisherman":            3,
    "flying-machine":       4,
    "freeze":               4,
    "furnace":              4,
    "giant":                5,
    "giant-skeleton":       6,
    "giant-snowball":       2,
    "goblin":               2,
    "goblin-barrel":        3,
    "goblin-cage":          4,
    "goblin-drill":         4,
    "goblin-giant":         6,
    "goblin-hut":           5,
    "golden-knight":        4,
    "golem":                8,
    "graveyard":            5,
    "guard":                3,
    "heal-spirit":          1,
    "hog-rider":            4,
    "hunter":               4,
    "ice-golem":            2,
    "ice-spirit":           1,
    "ice-spirit-evolution": 1,
    "ice-wizard":           3,
    "inferno-dragon":       4,
    "inferno-tower":        5,
    "knight":               3,
    "knight-evolution":     3,
    "lava-hound":           7,
    "lightning":            6,
    "little-prince":        3,
    "lumberjack":           4,
    "magic-archer":         4,
    "mega-knight":          7,
    "mega-minion":          3,
    "mighty-miner":         4,
    "mini-pekka":           4,
    "miner":                3,
    "minion":               3,
    "mirror":               None,
    "monk":                 5,
    "mortar":               4,
    "mortar-evolution":     4,
    "mother-witch":         4,
    "musketeer":            4,
    "night-witch":          4,
    "pekka":                7,
    "phoenix-big":          4,
    "poison":               4,
    "prince":               5,
    "princess":             3,
    "rage":                 2,
    "ram-rider":            5,
    "rascal-boy":           5,
    "rascal-girl":          5,
    "rocket":               6,
    "royal-delivery":       3,
    "royal-ghost":          3,
    "royal-giant":          6,
    "royal-giant-evolution":6,
    "royal-guardian":       4,
    "royal-hog":            5,
    "royal-recruit":        7,
    "royal-recruit-evolution":7,
    "skeleton":             1,
    "skeleton-barrel":      3,
    "skeleton-dragon":      4,
    "skeleton-evolution":   1,
    "skeleton-king":        4,
    "sparky":               6,
    "spear-goblin":         2,
    "tesla":                4,
    "tesla-evolution":      4,
    "the-log":              2,
    "tombstone":            3,
    "tornado":              3,
    "valkyrie":             4,
    "valkyrie-evolution":   4,
    "wall-breaker":         2,
    "wall-breaker-evolution":2,
    "witch":                5,
    "wizard":               5,
    "x-bow":                6,
    "zap":                  2,
    "zap-evolution":        2,
    "zappy":                4,
}
# fmt: on

# Sorted list used as the model vocabulary
CARD_NAMES = sorted(ELIXIR_COSTS.keys())
IDX_TO_CARD: Dict[int, str] = dict(enumerate(CARD_NAMES))
CARD_TO_IDX: Dict[str, int] = {name: idx for idx, name in IDX_TO_CARD.items()}
VOCAB_SIZE = len(CARD_NAMES)


def elixir_cost(card_name: str) -> Optional[int]:
    """Return elixir cost for a canonical card name, or None if unknown."""
    clean = card_name.lower().strip()
    return ELIXIR_COSTS.get(clean)


def card_to_idx(card_name: str) -> int:
    """Return vocabulary index (0-based) for a card name, or 0 if unknown."""
    clean = card_name.lower().strip()
    return CARD_TO_IDX.get(clean, 0)

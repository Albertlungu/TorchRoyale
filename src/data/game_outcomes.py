"""
Game outcome labels for TorchRoyale training data.

Hardcoded labels for games 1-23. Each game video should be labeled
as a Win ("W") or Loss ("L") for training the Decision Transformer.

Usage:
    from src.data.game_outcomes import GAME_OUTCOMES
    outcome = GAME_OUTCOMES.get("Game_1.mp4")  # Returns "W" or "L"
"""

# Game outcomes based on user-provided labels:
# 1-23: Win
# 4: L
# 5-6: W
# 7: L
# 8 onward: W

GAME_OUTCOMES = {
    "Game_1.mp4": "W",
    "Game_2.mp4": "W",
    "Game_3.mp4": "W",
    "Game_4.mp4": "L",
    "Game_5.mp4": "W",
    "Game_6.mp4": "W",
    "Game_7.mp4": "L",
    "Game_8.mp4": "W",
    "Game_9.mp4": "W",
    "Game_10.mp4": "W",
    "Game_11.mp4": "W",
    "Game_12.mp4": "W",
    "Game_13.mp4": "W",
    "Game_14.mp4": "W",
    "Game_15.mp4": "W",
    "Game_16.mp4": "W",
    "Game_17.mp4": "W",
    "Game_18.mp4": "W",
    "Game_19.mp4": "W",
    "Game_20.mp4": "W",
    "Game_21.mp4": "W",
    "Game_22.mp4": "W",
    "Game_23.mp4": "W",
}


def get_outcome(video_name: str) -> str | None:
    """
    Get the outcome for a game video.

    Args:
        video_name: Name of the video file (e.g., "Game_1.mp4").

    Returns:
        "W" for win, "L" for loss, or None if not labeled.
    """
    return GAME_OUTCOMES.get(video_name)

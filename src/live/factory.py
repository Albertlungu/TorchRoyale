"""Factory helpers for wiring the live TorchRoyale runtime into the UI."""

from typing import Any
from typing import Dict
from typing import Optional

from src.live.bot import TorchRoyaleBot


def create_torchroyale_bot(
    actions: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    log_handler=None,
):
    """
    Build the live TorchRoyale bot expected by the UI.
    
    Args:
        actions (Optional[Any]): Unused; actions are derived from config
        config (Optional[Dict[str, Any]]): Configuration for device and gameplay settings
        log_handler (Optional[Callable[[str], None]]): Callback function for UI logging
    Returns:
        (TorchRoyaleBot): Initialized bot instance
    """
    del actions
    if config is None:
        config = {}
    return TorchRoyaleBot(config=config, log_handler=log_handler)

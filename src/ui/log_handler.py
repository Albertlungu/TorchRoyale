"""
Log handler for displaying log messages in the GUI.
"""

import logging

from PyQt6.QtCore import Q_ARG
from PyQt6.QtCore import QMetaObject
from PyQt6.QtCore import Qt


class QTextEditLogger(logging.Handler):
    """Logging handler that outputs to a QTextEdit widget."""

    def __init__(self, text_edit) -> None:
        """Initialize the handler with a QTextEdit widget."""
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record) -> None:
        """Emit a log record to the text edit widget."""
        log_entry = self.format(record)
        QMetaObject.invokeMethod(
            self.text_edit,
            "append",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, log_entry),
        )

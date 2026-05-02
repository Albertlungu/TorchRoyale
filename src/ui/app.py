"""TorchRoyale desktop application bootstrap."""

import sys

from PyQt6.QtWidgets import QApplication

from src.live.factory import create_torchroyale_bot
from src.ui.main_window import MainWindow
from src.ui.utils import load_config


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(load_config(), bot_factory=create_torchroyale_bot)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""
Compatibility wrapper for the unified Windows YOLO26 entry point.

Prefer:
  py -3 scripts\\train_roboflow_yolo26_windows.py infer --source ...
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("train_roboflow_yolo26_windows.py")
    argv = [sys.executable, str(target), "infer", *sys.argv[1:]]
    raise SystemExit(__import__("subprocess").call(argv))


if __name__ == "__main__":
    main()

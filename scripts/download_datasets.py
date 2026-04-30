"""
Download YOLOv8 datasets from Roboflow for the dual-model detector.

Downloads two datasets:
  1. Cicadas dataset (player Hog 2.6 cards)
  2. Vision Bot all-enemy-cards dataset (opponent cards)

Each dataset is saved to its respective directory under data/datasets/.
If a data.yaml file already exists in the destination, the download is skipped.

Environment:
  ROBOFLOW_API_KEY must be set in .env file

Usage:
  python scripts/download_datasets.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    sys.exit(1)

try:
    from roboflow import Roboflow  # type: ignore
except ImportError:
    print("roboflow not installed. Install with: pip install roboflow")
    sys.exit(1)

# Project root and paths
_PROJECT_ROOT = Path(__file__).parents[1]
_DATA_DIR = _PROJECT_ROOT / "data" / "datasets"

# Dataset configurations: (workspace, project, version, destination)
_DATASETS: Tuple[Tuple[str, str, int, Path], ...] = (
    ("cicadas", "clash-royale-9eug2", 1, _DATA_DIR / "cicadas"),
    (
        "vision-bot",
        "clash-royale-all-enemy-cards-w9haz",
        1,
        _DATA_DIR / "visionbot_enemy",
    ),
)


def _print_stats(dataset_dir: Path, name: str) -> None:
    """
    Print dataset statistics from data.yaml and image counts.

    Args:
        dataset_dir (Path): path containing the dataset with data.yaml.
        name (str): human-readable dataset name for display.
    """
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"  [{name}] Warning: data.yaml not found")
        return

    # Parse yaml manually to avoid pyyaml dependency
    content = data_yaml.read_text()
    num_classes = 0
    class_names: list[str] = []
    num_train = 0
    num_valid = 0
    num_test = 0

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("nc:"):
            num_classes = int(line.split(":")[1].strip())
        elif line.startswith("names:"):
            # Simple parsing - assumes names follow on next lines with - prefix
            pass
        elif line.startswith("- "):
            class_names.append(line[2:].strip())
        elif "train:" in line:
            train_path = line.split(":")[1].strip()
            train_dir = dataset_dir / train_path
            if train_dir.exists():
                num_train = len(list(train_dir.glob("*.jpg"))) + len(
                    list(train_dir.glob("*.png"))
                )
        elif "valid:" in line:
            valid_path = line.split(":")[1].strip()
            valid_dir = dataset_dir / valid_path
            if valid_dir.exists():
                num_valid = len(list(valid_dir.glob("*.jpg"))) + len(
                    list(valid_dir.glob("*.png"))
                )
        elif "test:" in line:
            test_path = line.split(":")[1].strip()
            test_dir = dataset_dir / test_path
            if test_dir.exists():
                num_test = len(list(test_dir.glob("*.jpg"))) + len(
                    list(test_dir.glob("*.png"))
                )

    if not class_names:
        class_names = ["unknown"] * num_classes

    print(f"  [{name}] Classes: {num_classes}")
    print(
        f"  [{name}] Class names: {', '.join(class_names[:10])}"
        + ("..." if len(class_names) > 10 else "")
    )
    print(f"  [{name}] Train images: {num_train}")
    print(f"  [{name}] Valid images: {num_valid}")
    print(f"  [{name}] Test images: {num_test}")


def download_dataset(
    rf: Roboflow,
    workspace: str,
    project: str,
    version: int,
    destination: Path,
    format: str = "yolov8",
) -> None:
    """
    Download a single dataset from Roboflow.

    Args:
        rf (Roboflow): initialized Roboflow client.
        workspace (str): Roboflow workspace name.
        project (str): Roboflow project name.
        version (int): dataset version number.
        destination (Path): local directory to save the dataset.
        format (str): export format (default: yolov8).
    """
    dataset_name = f"{workspace}/{project}"
    print(f"Processing {dataset_name} v{version}...")

    # Skip if already downloaded
    if (destination / "data.yaml").exists():
        print(f"  Skipped: data.yaml already exists in {destination}")
        _print_stats(destination, dataset_name)
        return

    print(f"  Downloading to {destination}...")
    destination.mkdir(parents=True, exist_ok=True)

    try:
        rf_workspace = rf.workspace(workspace)
        rf_project = rf_workspace.project(project)
        rf_version = rf_project.version(version)

        # Download and extract
        dataset = rf_version.download(format, location=str(destination))
        print(f"  Downloaded successfully to {destination}")

    except Exception as e:
        print(f"  Error downloading {dataset_name}: {e}")
        return

    _print_stats(destination, dataset_name)


def main() -> None:
    """Download all configured datasets from Roboflow."""
    # Load API key from .env
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found. Using environment variables.")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in environment or .env file.")
        print("Please add ROBOFLOW_API_KEY=your_key_here to your .env file.")
        sys.exit(1)

    print("Initializing Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Roboflow: {e}")
        sys.exit(1)

    print("-" * 50)
    for workspace, project, version, dest in _DATASETS:
        download_dataset(rf, workspace, project, version, dest)
        print("-" * 50)

    print("All downloads complete.")


if __name__ == "__main__":
    main()

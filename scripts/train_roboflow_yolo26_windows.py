"""
Single-entry Windows script for training and inference with a YOLO26 nano enemy-card detector.

It is designed around the Roboflow Universe project:
https://universe.roboflow.com/stuff-j62hv/clash-royale-all-enemy-cards

Features:
1. Downloads the Roboflow dataset export locally.
2. Rebuilds the dataset at a stretched 640x640 resolution.
3. Creates exactly 2 augmented train outputs per original train image.
4. Trains YOLO26 nano with CUDA by default when available.
5. Exports ONNX and copies stable artifacts into data/models/onfield/.
6. Runs inference from the same script with `train` and `infer` subcommands.

Examples:
  py -3 scripts\\train_roboflow_yolo26_windows.py train --api-key YOUR_ROBOFLOW_KEY
  py -3 scripts\\train_roboflow_yolo26_windows.py infer --source C:\\images\\frame.png
  py -3 scripts\\train_roboflow_yolo26_windows.py infer --weights data\\models\\onfield\\clash-royale-all-enemy-cards-yolo26n-best.pt --source 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "datasets" / "rf_clash_royale_enemy_cards"
DEFAULT_RUNS_PROJECT = PROJECT_ROOT / "runs" / "train"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "onfield"
DEFAULT_CANONICAL_STEM = "clash-royale-all-enemy-cards-yolo26n-best"
DEFAULT_PT_WEIGHTS = DEFAULT_MODEL_DIR / f"{DEFAULT_CANONICAL_STEM}.pt"
DEFAULT_ONNX_WEIGHTS = DEFAULT_MODEL_DIR / f"{DEFAULT_CANONICAL_STEM}.onnx"
DEFAULT_LATEST_SUMMARY = DEFAULT_MODEL_DIR / "clash-royale-all-enemy-cards-yolo26n-latest.json"
TARGET_SIZE = 640


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("opencv-python is required. Install with: pip install opencv-python") from exc
    return cv2


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc
    return yaml


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit("numpy is required. Install with: pip install numpy") from exc
    return np


def _require_roboflow() -> Any:
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError as exc:
        raise SystemExit("roboflow is required. Install with: pip install roboflow") from exc
    return Roboflow


def _require_yolo() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from exc
    return YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-entry Windows YOLO26 trainer/inference runner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Download, preprocess, and train YOLO26.")
    train_parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    train_parser.add_argument("--workspace", default="stuff-j62hv")
    train_parser.add_argument("--project", default="clash-royale-all-enemy-cards")
    train_parser.add_argument("--version", type=int, default=1)
    train_parser.add_argument("--format", default="yolo26")
    train_parser.add_argument("--model", default="yolo26n.pt")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--workers", type=int, default=8)
    train_parser.add_argument("--output-root", default=str(DEFAULT_DATA_ROOT))
    train_parser.add_argument("--runs-project", default=str(DEFAULT_RUNS_PROJECT))
    train_parser.add_argument("--run-name", default="clash_royale_enemy_yolo26n")
    train_parser.add_argument("--canonical-stem", default=DEFAULT_CANONICAL_STEM)
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--redownload", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Run inference with trained weights.")
    infer_parser.add_argument(
        "--weights",
        default="",
        help="Path to .pt or .onnx weights. Defaults to the canonical best weights if present.",
    )
    infer_parser.add_argument("--source", required=True, help="Image, folder, video path, or webcam index.")
    infer_parser.add_argument("--conf", type=float, default=0.25)
    infer_parser.add_argument("--iou", type=float, default=0.5)
    infer_parser.add_argument("--imgsz", type=int, default=TARGET_SIZE)
    infer_parser.add_argument("--device", default="auto")
    infer_parser.add_argument("--save-dir", default="output/infer_yolo26")
    infer_parser.add_argument("--show", action="store_true")

    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML content in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    yaml = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return _detect_device()


def _resolve_split_dir(base_dir: Path, raw_value: str) -> Path:
    split_dir = (base_dir / raw_value).resolve()
    if split_dir.exists():
        return split_dir
    fallback = (base_dir / raw_value.replace("../", "")).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve split path '{raw_value}' from {base_dir}")


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
    destination: Path,
    redownload: bool,
) -> Path:
    data_yaml = destination / "data.yaml"
    if data_yaml.exists() and not redownload:
        print(f"[download] Reusing existing dataset at {destination}")
        return destination

    if destination.exists() and redownload:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"[download] Downloading {workspace}/{project}/{version} as {export_format} ...")
    Roboflow = _require_roboflow()
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(export_format, location=str(destination))
    )
    downloaded_location = Path(str(dataset.location))
    if downloaded_location != destination and downloaded_location.exists():
        destination = downloaded_location
    if not (destination / "data.yaml").exists():
        raise FileNotFoundError(f"Downloaded dataset is missing data.yaml under {destination}")
    return destination


def _apply_hsv(image: Any, rng: random.Random) -> Any:
    cv2 = _require_cv2()
    np = _require_numpy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_delta_degrees = rng.uniform(-11.0, 11.0)
    sat_scale = 1.0 + rng.uniform(-0.25, 0.25)
    val_scale = 1.0 + rng.uniform(-0.15, 0.15)

    hsv[..., 0] = (hsv[..., 0] + (hue_delta_degrees / 2.0)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_blur(image: Any, rng: random.Random) -> Any:
    cv2 = _require_cv2()
    sigma = rng.uniform(0.0, 0.4)
    if sigma <= 0.01:
        return image
    return cv2.GaussianBlur(image, (3, 3), sigmaX=sigma, sigmaY=sigma)


def _apply_noise(image: Any, rng: random.Random) -> Any:
    np = _require_numpy()
    noisy = image.copy()
    h, w = noisy.shape[:2]
    total_pixels = h * w
    pixel_fraction = rng.uniform(0.0, 0.0032)
    count = max(0, int(total_pixels * pixel_fraction))
    if count == 0:
        return noisy

    ys = np.array([rng.randrange(h) for _ in range(count)], dtype=np.int32)
    xs = np.array([rng.randrange(w) for _ in range(count)], dtype=np.int32)
    colors = np.array(
        [[rng.randrange(256), rng.randrange(256), rng.randrange(256)] for _ in range(count)],
        dtype=np.uint8,
    )
    noisy[ys, xs] = colors
    return noisy


def augment_image(image: Any, rng: random.Random) -> Any:
    augmented = _apply_hsv(image, rng)
    augmented = _apply_blur(augmented, rng)
    augmented = _apply_noise(augmented, rng)
    return augmented


def _copy_label(src_label: Path, dst_label: Path) -> None:
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def build_processed_dataset(raw_dir: Path, output_dir: Path, seed: int) -> Path:
    cv2 = _require_cv2()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_yaml = _load_yaml(raw_dir / "data.yaml")
    names = raw_yaml.get("names", [])
    if not isinstance(names, list):
        raise ValueError("Expected 'names' to be a list in data.yaml")

    split_key_map = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    normalized_splits: dict[str, str] = {}
    for raw_key, normalized_key in split_key_map.items():
        if raw_key in raw_yaml and normalized_key not in normalized_splits:
            normalized_splits[normalized_key] = str(raw_yaml[raw_key])

    for split_name, split_path in normalized_splits.items():
        image_dir = _resolve_split_dir(raw_dir, split_path)
        label_dir = image_dir.parent.parent / "labels"
        out_image_dir = output_dir / split_name / "images"
        out_label_dir = output_dir / split_name / "labels"
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        print(f"[prepare] {split_name}: {len(image_paths)} images")

        for index, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to load image: {image_path}")
            stretched = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            label_path = label_dir / f"{image_path.stem}.txt"

            base_name = image_path.stem
            ext = image_path.suffix.lower()
            dst_image = out_image_dir / f"{base_name}{ext}"
            dst_label = out_label_dir / f"{base_name}.txt"
            cv2.imwrite(str(dst_image), stretched)
            _copy_label(label_path, dst_label)

            if split_name != "train":
                continue

            for aug_idx in range(2):
                rng = random.Random(seed + (index * 101) + aug_idx)
                aug_image = augment_image(stretched, rng)
                aug_name = f"{base_name}__aug{aug_idx + 1}"
                aug_image_path = out_image_dir / f"{aug_name}{ext}"
                aug_label_path = out_label_dir / f"{aug_name}.txt"
                cv2.imwrite(str(aug_image_path), aug_image)
                _copy_label(label_path, aug_label_path)

    processed_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images" if (output_dir / "val" / "images").exists() else "valid/images",
        "test": "test/images" if (output_dir / "test" / "images").exists() else "",
        "names": names,
        "nc": len(names),
    }
    if not processed_yaml["test"]:
        processed_yaml.pop("test")

    processed_yaml_path = output_dir / "data.yaml"
    _write_yaml(processed_yaml_path, processed_yaml)
    metadata = {
        "target_size": TARGET_SIZE,
        "stretch": [TARGET_SIZE, TARGET_SIZE],
        "train_augmented_outputs_per_example": 2,
        "augmentations": {
            "hue_degrees": [-11, 11],
            "saturation_scale_delta": [-0.25, 0.25],
            "brightness_scale_delta": [-0.15, 0.15],
            "blur_sigma_px": [0.0, 0.4],
            "noise_pixel_fraction": [0.0, 0.0032],
        },
    }
    (output_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return processed_yaml_path


def train_and_export(
    model_name: str,
    dataset_yaml: Path,
    epochs: int,
    batch: int,
    device: str,
    workers: int,
    runs_project: Path,
    run_name: str,
    seed: int,
) -> tuple[Path, Path, Path]:
    print(f"[train] Loading model: {model_name}")
    print(f"[train] Using device: {device}")
    YOLO = _require_yolo()
    model = YOLO(model_name)
    results = model.train(
        data=str(dataset_yaml),
        imgsz=TARGET_SIZE,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        project=str(runs_project),
        name=run_name,
        pretrained=True,
        seed=seed,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training completed but best weights were not found: {best_weights}")

    print(f"[export] Exporting ONNX from {best_weights}")
    best_model = YOLO(str(best_weights))
    exported = best_model.export(format="onnx", imgsz=TARGET_SIZE)
    onnx_path = Path(str(exported))
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX export did not produce a file: {onnx_path}")
    return save_dir, best_weights, onnx_path


def publish_artifacts(
    best_weights: Path,
    onnx_path: Path,
    canonical_stem: str,
) -> tuple[Path, Path]:
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    canonical_pt = DEFAULT_MODEL_DIR / f"{canonical_stem}.pt"
    canonical_onnx = DEFAULT_MODEL_DIR / f"{canonical_stem}.onnx"
    shutil.copy2(best_weights, canonical_pt)
    shutil.copy2(onnx_path, canonical_onnx)
    print(f"[artifacts] Copied PT weights to: {canonical_pt}")
    print(f"[artifacts] Copied ONNX weights to: {canonical_onnx}")
    return canonical_pt, canonical_onnx


def write_latest_summary(summary: dict[str, Any]) -> None:
    DEFAULT_LATEST_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LATEST_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def resolve_default_weights(explicit_weights: str) -> Path:
    if explicit_weights:
        path = Path(explicit_weights)
        if not path.exists():
            raise SystemExit(f"Weights not found: {path}")
        return path

    if DEFAULT_PT_WEIGHTS.exists():
        return DEFAULT_PT_WEIGHTS

    if DEFAULT_LATEST_SUMMARY.exists():
        summary = json.loads(DEFAULT_LATEST_SUMMARY.read_text(encoding="utf-8"))
        candidate = Path(summary["canonical_pt_weights"])
        if candidate.exists():
            return candidate

    raise SystemExit(
        "No default weights found. Run the train command first or pass --weights explicitly."
    )


def run_inference(
    weights_path: Path,
    source_value: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    save_dir: str,
    show: bool,
) -> None:
    cv2 = _require_cv2()
    YOLO = _require_yolo()
    source: str | int
    if source_value.isdigit():
        source = int(source_value)
    else:
        source = source_value

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    predict_kwargs: dict[str, Any] = {
        "source": source,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "save": True,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
        "stream": False,
        "verbose": True,
    }
    if weights_path.suffix.lower() == ".pt":
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)

    count = 0
    for result in results:
        count += 1
        if show:
            frame = result.plot()
            cv2.imshow("YOLO26 Inference", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    if show:
        cv2.destroyAllWindows()

    print(f"Saved inference outputs to: {output_dir.resolve()}")
    print(f"Processed result items: {count}")


def handle_train(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("Missing Roboflow API key. Pass --api-key or set ROBOFLOW_API_KEY.")

    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    processed_dir = output_root / "processed_640"
    runs_project = Path(args.runs_project)

    raw_dataset_dir = download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        export_format=args.format,
        destination=raw_dir,
        redownload=args.redownload,
    )
    dataset_yaml = build_processed_dataset(raw_dataset_dir, processed_dir, args.seed)
    save_dir, best_weights, onnx_path = train_and_export(
        model_name=args.model,
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        workers=args.workers,
        runs_project=runs_project,
        run_name=args.run_name,
        seed=args.seed,
    )
    canonical_pt, canonical_onnx = publish_artifacts(
        best_weights=best_weights,
        onnx_path=onnx_path,
        canonical_stem=args.canonical_stem,
    )

    summary = {
        "workspace": args.workspace,
        "project": args.project,
        "version": args.version,
        "model": args.model,
        "device": device,
        "dataset_yaml": str(dataset_yaml.resolve()),
        "run_dir": str(save_dir.resolve()),
        "best_weights": str(best_weights.resolve()),
        "onnx_weights": str(onnx_path.resolve()),
        "canonical_pt_weights": str(canonical_pt.resolve()),
        "canonical_onnx_weights": str(canonical_onnx.resolve()),
    }
    summary_path = save_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_latest_summary(summary)

    print("\nTraining complete.")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Run directory: {save_dir}")
    print(f"Best weights: {best_weights}")
    print(f"ONNX weights: {onnx_path}")
    print(f"Canonical PT weights: {canonical_pt}")
    print(f"Canonical ONNX weights: {canonical_onnx}")
    print("Inference command:")
    print(
        "  py -3 scripts\\train_roboflow_yolo26_windows.py infer "
        f'--weights "{canonical_pt}" --source "C:\\path\\to\\image_or_video.jpg"'
    )


def handle_infer(args: argparse.Namespace) -> None:
    weights_path = resolve_default_weights(args.weights)
    device = resolve_device(args.device)
    print(f"[infer] Using weights: {weights_path}")
    print(f"[infer] Using device: {device}")
    run_inference(
        weights_path=weights_path,
        source_value=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save_dir=args.save_dir,
        show=args.show,
    )


def main() -> None:
    args = parse_args()
    if args.command == "train":
        handle_train(args)
        return
    if args.command == "infer":
        handle_infer(args)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

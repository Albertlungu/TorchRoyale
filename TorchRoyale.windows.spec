# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


project_root = Path.cwd()
datas = [
    (str(project_root / "configs"), "configs"),
    (str(project_root / "data" / "images"), "data/images"),
    (str(project_root / "data" / "models"), "data/models"),
    (str(project_root / "build_assets" / "platform-tools"), "platform-tools"),
]

binaries = collect_dynamic_libs("onnxruntime")

hiddenimports = [
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
]

excludes = [
    "pytest",
    "mkdocs",
    "pygments",
    "torch",
    "torchvision",
    "transformers",
    "tensorflow",
    "timm",
    "pandas",
    "matplotlib",
    "boto3",
    "botocore",
    "sklearn",
    "joblib",
    "cv2",
    "easyocr",
    "roboflow",
    "inference",
    "supervision",
    "ultralytics",
    "av",
    "h5py",
    "src.classifier",
    "src.data",
    "src.overlay",
    "src.ocr",
    "src.recommendation",
    "src.transformer",
    "src.video",
    "tests",
    "tests_debug",
]

a = Analysis(
    ["run_ui.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TorchRoyale",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TorchRoyale",
)

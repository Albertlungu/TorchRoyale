#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"
BUILD_ASSETS_DIR="$ROOT_DIR/build_assets"
PLATFORM_TOOLS_DIR="$BUILD_ASSETS_DIR/platform-tools"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing virtualenv python at $VENV_PYTHON" >&2
  exit 1
fi

resolve_platform_tools_dir() {
  local adb_path=""

  if command -v adb >/dev/null 2>&1; then
    adb_path="$(command -v adb)"
  elif [[ -x "${ANDROID_SDK_ROOT:-}/platform-tools/adb" ]]; then
    adb_path="${ANDROID_SDK_ROOT}/platform-tools/adb"
  elif [[ -x "${ANDROID_HOME:-}/platform-tools/adb" ]]; then
    adb_path="${ANDROID_HOME}/platform-tools/adb"
  elif [[ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]]; then
    adb_path="$HOME/Library/Android/sdk/platform-tools/adb"
  elif [[ -x "$HOME/Downloads/platform-tools/adb" ]]; then
    adb_path="$HOME/Downloads/platform-tools/adb"
  fi

  if [[ -z "$adb_path" ]]; then
    return 1
  fi

  dirname "$adb_path"
}

if ! "$VENV_PYTHON" -c "import PyInstaller" >/dev/null 2>&1; then
  "$VENV_PYTHON" -m pip install pyinstaller
fi

SOURCE_PLATFORM_TOOLS_DIR="$(resolve_platform_tools_dir || true)"
if [[ -z "$SOURCE_PLATFORM_TOOLS_DIR" || ! -d "$SOURCE_PLATFORM_TOOLS_DIR" ]]; then
  echo "Unable to locate Android platform-tools. Install adb or set ANDROID_SDK_ROOT." >&2
  exit 1
fi

rm -rf "$PLATFORM_TOOLS_DIR"
mkdir -p "$BUILD_ASSETS_DIR"
cp -R "$SOURCE_PLATFORM_TOOLS_DIR" "$PLATFORM_TOOLS_DIR"
chmod +x "$PLATFORM_TOOLS_DIR/adb"

cd "$ROOT_DIR"
rm -rf build dist
"$VENV_PYTHON" -m PyInstaller --noconfirm TorchRoyale.spec

echo
echo "Build complete:"
echo "  App bundle: $ROOT_DIR/dist/TorchRoyale.app"
echo "  Executable: $ROOT_DIR/dist/TorchRoyale"

#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="${1:-$ROOT_DIR/dist/TorchRoyale.app}"
BACKGROUND_SOURCE="${2:-/Users/ksukshavasi/Downloads/clash-royale-royal-chef-tower-troop-4273990514.jpg}"
OUTPUT_DMG="${3:-$ROOT_DIR/dist/TorchRoyale.dmg}"
VOLUME_NAME="TorchRoyale Installer"
WINDOW_WIDTH=720
WINDOW_HEIGHT=405
ICON_SIZE=128
APP_ICON_X=180
APP_ICON_Y=210
APPS_ICON_X=540
APPS_ICON_Y=210

if [[ ! -d "$APP_PATH" ]]; then
  echo "Missing app bundle at $APP_PATH" >&2
  exit 1
fi

if [[ ! -f "$BACKGROUND_SOURCE" ]]; then
  echo "Missing background image at $BACKGROUND_SOURCE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DMG")"

WORK_DIR="$(mktemp -d /private/tmp/torchroyale-dmg.XXXXXX)"
STAGING_DIR="$WORK_DIR/staging"
BACKGROUND_DIR="$STAGING_DIR/.background"
RESIZED_BACKGROUND="$BACKGROUND_DIR/background.png"
RW_DMG="$WORK_DIR/TorchRoyale-temp.dmg"
MOUNT_OUTPUT="$WORK_DIR/mount.txt"

cleanup() {
  if [[ -n "${DEVICE_NAME:-}" ]]; then
    hdiutil detach "$DEVICE_NAME" -quiet || true
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$BACKGROUND_DIR"
cp -R "$APP_PATH" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

sips -s format png "$BACKGROUND_SOURCE" --resampleHeightWidth "$WINDOW_HEIGHT" "$WINDOW_WIDTH" --out "$RESIZED_BACKGROUND" >/dev/null
SetFile -a V "$BACKGROUND_DIR"

rm -f "$OUTPUT_DMG"

hdiutil create \
  -srcfolder "$STAGING_DIR" \
  -volname "$VOLUME_NAME" \
  -fs HFS+ \
  -fsargs "-c c=64,a=16,e=16" \
  -format UDRW \
  "$RW_DMG" >/dev/null

hdiutil attach -readwrite -noverify -noautoopen "$RW_DMG" >"$MOUNT_OUTPUT"
DEVICE_NAME="$(awk '/Apple_HFS/ {print $1}' "$MOUNT_OUTPUT" | head -n 1)"
VOLUME_PATH="/Volumes/$VOLUME_NAME"

if [[ -z "$DEVICE_NAME" || ! -d "$VOLUME_PATH" ]]; then
  echo "Failed to mount temporary DMG." >&2
  exit 1
fi

osascript <<EOF >/dev/null
tell application "Finder"
  tell disk "$VOLUME_NAME"
    open
    tell container window
      set current view to icon view
      set toolbar visible to false
      set statusbar visible to false
      set bounds to {100, 100, 100 + $WINDOW_WIDTH, 100 + $WINDOW_HEIGHT}
    end tell
    tell icon view options of container window
      set icon size to $ICON_SIZE
      set text size to 14
      set arrangement to not arranged
      set background picture to (POSIX file "$VOLUME_PATH/.background/background.png" as alias)
    end tell
    set position of item "TorchRoyale.app" to {$APP_ICON_X, $APP_ICON_Y}
    set position of item "Applications" to {$APPS_ICON_X, $APPS_ICON_Y}
    close
    open
    update without registering applications
    delay 2
  end tell
end tell
EOF

sync
hdiutil detach "$DEVICE_NAME" -quiet
unset DEVICE_NAME

hdiutil convert "$RW_DMG" -format UDZO -imagekey zlib-level=9 -o "$OUTPUT_DMG" >/dev/null

echo "Created DMG at $OUTPUT_DMG"

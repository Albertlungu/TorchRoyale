#!/bin/bash
set -e

echo "Building TorchRoyale.app locally..."

# Clean previous build
rm -rf python pbs.tar.gz TorchRoyale.app dmg-stage TorchRoyale-macOS.dmg

# Fetch self-contained Python (no external framework dependencies)
curl -L "https://github.com/indygreg/python-build-standalone/releases/download/20250127/cpython-3.12.9+20250127-aarch64-apple-darwin-install_only.tar.gz" -o pbs.tar.gz
tar -xf pbs.tar.gz
rm pbs.tar.gz

python/bin/pip3 install --upgrade pip -q
python/bin/pip3 install -r requirements.txt -q

# Assemble .app bundle
mkdir -p TorchRoyale.app/Contents/MacOS
mkdir -p TorchRoyale.app/Contents/Resources

cat > TorchRoyale.app/Contents/MacOS/TorchRoyale << 'EOF'
#!/bin/bash
RESOURCES="$(cd "$(dirname "$0")/../Resources" && pwd)"
exec "$RESOURCES/python-dist/bin/python3" "$RESOURCES/run_ui.py"
EOF
chmod +x TorchRoyale.app/Contents/MacOS/TorchRoyale

cat > TorchRoyale.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>TorchRoyale</string>
  <key>CFBundleExecutable</key><string>TorchRoyale</string>
  <key>CFBundleIdentifier</key><string>com.torchroyale.app</string>
  <key>CFBundleVersion</key><string>1.0.0</string>
  <key>CFBundleShortVersionString</key><string>1.0.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
</dict>
</plist>
EOF

cp run_ui.py TorchRoyale.app/Contents/Resources/
cp run_inference.py TorchRoyale.app/Contents/Resources/
cp -r src TorchRoyale.app/Contents/Resources/
cp -r data TorchRoyale.app/Contents/Resources/
cp -r configs TorchRoyale.app/Contents/Resources/
cp *.pt TorchRoyale.app/Contents/Resources/ 2>/dev/null || true
cp -r python TorchRoyale.app/Contents/Resources/python-dist

echo "TorchRoyale.app assembled."

# Optionally wrap in DMG
if command -v create-dmg &> /dev/null; then
    mkdir dmg-stage
    cp -r TorchRoyale.app dmg-stage/TorchRoyale.app
    create-dmg \
        --volname "TorchRoyale" \
        --window-size 800 400 \
        --icon-size 128 \
        --icon "TorchRoyale.app" 200 200 \
        --app-drop-link 600 200 \
        "TorchRoyale-macOS.dmg" \
        dmg-stage
    echo "TorchRoyale-macOS.dmg created."
else
    echo "create-dmg not found — skipping DMG. Install with: brew install create-dmg"
fi

echo "Done. To install: cp -r TorchRoyale.app /Applications/TorchRoyale.app"

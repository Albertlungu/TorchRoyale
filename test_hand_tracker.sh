#!/bin/bash
# Test the HandTracker branch on Game 2
# Run this in one terminal while test_vlm.sh runs in another

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="/Users/albertlungu/Local/GitHub/TorchRoyale/venv/bin/python3"
mkdir -p "$REPO/logs"

echo "=== Branch: feature/hand-tracker ==="
echo "=== Patching hand_cards using HandTracker ==="

$VENV scripts/patch_hand_cards.py \
    --analyses-dir output/analysis \
    2>&1 | tee logs/test_hand_tracker.log

echo ""
echo "=== Sample patched hand_cards from Game 2 (frames 50-60) ==="
$VENV - <<'EOF'
import json
with open('output/analysis/Game 2_analysis.json') as f:
    d = json.load(f)
for fr in d['frames'][50:60]:
    print(f"  t={fr['timestamp_ms']}ms  hand={fr.get('hand_cards', [])}")
EOF

echo ""
echo "=== Running inference on Game 2 ==="
$VENV run_inference.py "data/replays/Game 2.mov" \
    --checkpoint data/models/dt/best.pt \
    2>&1 | tee -a logs/test_hand_tracker.log

echo ""
echo "=== Opening viewer ==="
$VENV view_replay.py \
    "data/replays/Game 2.mov" \
    "output/replay_runs/Game 2_recommendations.jsonl"

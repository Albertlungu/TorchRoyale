#!/bin/bash
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
MAIN="/Users/albertlungu/Local/GitHub/TorchRoyale"
VENV="$MAIN/venv/bin/python3"

mkdir -p "$MAIN/logs"
cd "$MAIN"

echo "=== Branch: feature/hand-tracker ==="
echo "=== Re-running fix_onfield (fixes -next cards marked as is_on_field=True) ==="

$VENV "$REPO/scripts/fix_onfield_classifications.py" \
    "$MAIN/output/analysis/Game 1_analysis.json" \
    "$MAIN/output/analysis/Game 2_analysis.json" \
    "$MAIN/output/analysis/Game 3_analysis.json" \
    "$MAIN/output/analysis/Game 4_analysis.json" \
    "$MAIN/output/analysis/Game 5_analysis.json" \
    "$MAIN/output/analysis/Game 6_analysis.json" \
    "$MAIN/output/analysis/Game 7_analysis.json" \
    "$MAIN/output/analysis/Game 8_analysis.json" \
    "$MAIN/output/analysis/Game 9_analysis.json" \
    2>&1 | tee "$REPO/logs/test_hand_tracker.log"

echo ""
echo "=== Patching hand_cards using HandTracker ==="

$VENV "$REPO/scripts/patch_hand_cards.py" \
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

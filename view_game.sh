#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./view_game.sh <game_number>"
    echo "Example: ./view_game.sh 1"
    exit 1
fi

GAME="Game $1"
VIDEO="data/replays/${GAME}.mov"
ANALYSIS="output/analysis/${GAME}_analysis.json"
CHECKPOINT="data/models/dt/best.pt"
JSONL="output/replay_runs/${GAME}_recommendations.jsonl"

if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

echo "=== Running inference for $GAME ==="
venv/bin/python3 run_inference.py "$VIDEO" \
    --checkpoint "$CHECKPOINT"

echo "=== Opening replay viewer ==="
venv/bin/python3 view_replay.py "$VIDEO" "$JSONL"

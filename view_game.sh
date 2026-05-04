#!/bin/bash
set -e
if [ -z "$1" ]; then echo "Usage: ./view_game.sh <number>"; exit 1; fi

GAME="Game $1"
VIDEO="data/replays/${GAME}.mov"
CHECKPOINT="data/models/dt/best.pt"
JSONL="output/replay_runs/${GAME}_recommendations.jsonl"

if [ ! -f "$VIDEO" ]; then echo "ERROR: $VIDEO not found"; exit 1; fi

echo "=== Running inference for $GAME ==="
venv/bin/python3 run_inference.py "$VIDEO" --checkpoint "$CHECKPOINT"

echo "=== Opening viewer ==="
venv/bin/python3 view_replay.py "$VIDEO" "$JSONL"

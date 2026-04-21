#!/bin/bash
# Run inference on a replay video and then display it with the overlay
#
# Usage:
#   ./run_inference.sh                           # Auto-detect video in data/replays/
#   ./run_inference.sh path/to/video.mp4         # Specific video
#   ./run_inference.sh --frame-skip 120          # With frame skip
#   ./run_inference.sh --no-cache                # Force re-analysis
#   ./run_inference.sh --cached-analysis path    # Use specific cached analysis

set -e  # Exit on error

# Default values
VIDEO=""
FRAME_SKIP=6
CHECKPOINT="output/models/best.pt"
CACHE_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --frame-skip)
            FRAME_SKIP="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --no-cache)
            CACHE_ARG="--no-cache"
            shift
            ;;
        --cached-analysis)
            CACHE_ARG="--cached-analysis $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_inference.sh [VIDEO] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  VIDEO                        Path to video file (default: auto-detect in data/replays/)"
            echo ""
            echo "Options:"
            echo "  --frame-skip SKIP            Frame skip value (default: 6)"
            echo "  --checkpoint PATH            Model checkpoint (default: output/models/best.pt)"
            echo "  --no-cache                   Force re-analysis, ignore cached analysis"
            echo "  --cached-analysis PATH       Use specific cached analysis JSON file"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_inference.sh                                    # Auto-detect video, frame-skip=6"
            echo "  ./run_inference.sh --frame-skip 120                   # High frame skip"
            echo "  ./run_inference.sh data/replays/video.mp4 --no-cache  # Force re-analysis"
            exit 0
            ;;
        *)
            if [[ -z "$VIDEO" ]]; then
                VIDEO="$1"
            else
                echo "ERROR: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Auto-detect video if not provided
if [[ -z "$VIDEO" ]]; then
    echo "No video specified, searching data/replays/..."
    VIDEO=$(find data/replays -name "*.mp4" -o -name "*.MP4" -o -name "*.mov" -o -name "*.MOV" | head -1)

    if [[ -z "$VIDEO" ]]; then
        echo "ERROR: No video found in data/replays/"
        exit 1
    fi

    echo "Found video: $VIDEO"
fi

# Check video exists
if [[ ! -f "$VIDEO" ]]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

# Extract video filename without extension
VIDEO_BASENAME=$(basename "$VIDEO")
VIDEO_STEM="${VIDEO_BASENAME%.*}"

# Output paths
OUTPUT_JSONL="output/replay_runs/${VIDEO_STEM}_recommendations.jsonl"

echo "========================================"
echo "Running Inference Pipeline"
echo "========================================"
echo "Video: $VIDEO"
echo "Frame skip: $FRAME_SKIP"
echo "Checkpoint: $CHECKPOINT"
if [[ -n "$CACHE_ARG" ]]; then
    echo "Cache: $CACHE_ARG"
fi
echo "Output: $OUTPUT_JSONL"
echo ""

# Run inference
python3 run_inference.py "$VIDEO" \
    --frame-skip "$FRAME_SKIP" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_JSONL" \
    $CACHE_ARG

# Check if overlay/replay viewer exists
if [[ -f "src/overlay/replay_runner.py" ]] || [[ -f "src/overlay/video_player.py" ]]; then
    echo ""
    echo "========================================"
    echo "Inference Complete!"
    echo "========================================"
    echo "Recommendations: $OUTPUT_JSONL"
    echo ""
    echo "To view the replay with overlay, run:"
    echo "  python3 view_replay.py \"$VIDEO\" \"$OUTPUT_JSONL\""
    echo ""
else
    echo ""
    echo "========================================"
    echo "Inference Complete!"
    echo "========================================"
    echo "Recommendations written to: $OUTPUT_JSONL"
    echo ""
fi

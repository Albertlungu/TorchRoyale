# TorchRoyale: Clash Royale AI Training Pipeline

A complete machine learning pipeline for training a Decision Transformer model to play Clash Royale. The system analyzes gameplay videos, detects cards and units in real-time, tracks hand state across frames, and generates training episodes for a causal transformer model.

## Architecture Overview

```
Replay Videos → Frame Analysis → Episode Generation → Model Training
    ↓              ↓                   ↓                   ↓
 .mp4 files   JSON state per    Trajectory           Decision
              frame           collections        Transformer
```

### Three-Layer Pipeline

1. **Detection Layer** (`src/detection/`)
   - KataCR: Dual YOLOv8 models for on-field unit detection
   - HandClassifier: YOLOv8 classification model for 4-card hand slots
   - HandTracker: Stateful tracker that stabilizes hand state and detects evolution

2. **Analysis Layer** (`src/overlay/`)
   - VideoAnalyzer: Orchestrates detection across all frames
   - Applies post-processing rules (hero musketeer substitution, evolution labeling)
   - Outputs per-frame state JSON with detections, hand cards, timers, elixir

3. **Training Layer** (`src/transformer/`)
   - Feature encoder: Converts frame state to fixed-length vectors
   - Decision Transformer: GPT-style causal transformer predicting card + placement
   - Training loop with validation and checkpoint saving

---

## Core Components

### Detection Pipeline

#### KataCRDetector (`src/detection/katacr.py`)

Wraps two YOLOv8 models from KataCR v0.7.13 for unit detection.

**Key features:**
- Dual-model combo with NMS merging for robust detections
- Automatic model weight download from Google Drive on first use
- Coordinate transformation from part2 crop (576×896) to full frame
- Motion-based opponent/player classification using tile_y movement

**Model outputs per detection:**
```
[x1, y1, x2, y2, confidence, class_idx]
```

Coordinates are in the 576×896 arena crop (part2), automatically transformed to full-frame and then to tile coordinates.

**Noise filtering:**
Removes KataCR internal classes that aren't real game units:
- `padding_belong`: Arena ownership marker (no actual unit)
- `bar`, `bar-level`: Health/elixir bars
- `clock`, `elixir`: UI elements
- `emote`: Cosmetics
- `evolution-symbol`: Evo UI indicator
- `tower`: Defensive structures (handled separately)

**Motion-based ownership correction:**

Units are initially classified as opponent/player based on spawn position (`tile_y < PLAYER_SIDE_MIN_ROW`). However, units can cross the river or be placed in the opponent's half after a tower break. The motion tracker compares unit positions across snapshots (every 2 frames) and corrects ownership based on vertical movement:

- `tile_y increased` → moving toward player side (bottom) → opponent unit
- `tile_y decreased` → moving toward opponent side (top) → player unit
- `tile_y unchanged` → static unit keeps spawn-based label

This logic only applies to units in the active play zone (rows 2–29), allowing king tower rows to use spawn-based labels.

**Calibration:**

The detector auto-calibrates by scanning a mid-game frame (10% into video) to find black letterbox edges and determine the game content bounds. Aspect ratio bucketing selects appropriate part2 crop parameters for portrait-in-landscape layouts.

#### HandClassifier (`src/detection/hand_classifier.py`)

YOLOv8 classification model that identifies cards in four hand slots.

**Layout:**
- Crops vertical strip: 84.5%–96.5% of frame height
- Divides into 5 sections: "next card" preview + 4 hand slots
- Applies per-slot horizontal offsets (calibrated for 1920×1080, scaled to any resolution)

**Output normalization:**

Raw classifier output is normalized to pipeline canonical format:
- Spaces → dashes: `"ice spirit"` → `"ice-spirit"`
- Plural → singular: `"skeletons"` → `"skeleton"`
- "evo " prefix → "-evolution" suffix: `"evo ice spirit"` → `"ice-spirit-evolution"`

This ensures downstream consumers (vocabulary lookup, hand tracker) see consistent names.

**Confidence threshold:** 0.40 (returns None for lower confidence)

#### HandTracker (`src/detection/hand_tracker.py`)

Stateful tracker that maintains a stable 4-card hand state and detects card evolutions.

**Forward-filling:**

Roboflow detects fewer than 4 cards per frame (~10% frame coverage). HandTracker forward-fills from the last known state and updates only when new classifications appear.

**Card play detection:**

Plays are detected when a card that was in-hand appears on-field:
- Tracks `(card_name, tile_x, tile_y)` tuples to distinguish replays at different positions
- Removes card from hand when detected on-field
- Increments play cycle counter

**Evolution tracking state machine:**

For candidates (`ice-spirit`, `skeleton`, `cannon`, `musketeer`):

1. **First play**: Card starts with `_evo_status=None` (unknown)
2. **Cycle count increments**: Each time card is played, counter increases
3. **Cycle == 2, status unknown**: Runs hand classifier once to confirm evolution
4. **Status resolved**: Stores `True` (has evo) or `False` (normal forever)
5. **Cycle == 2, status confirmed**: Labels card as `"{card}-evolution-in-hand"` in output
6. **Cycle resets**: Counter goes to 0 after output, ready for next 2-cycle

Output format:
- Normal card: `"{card}-in-hand"`
- Evolved card at cycle 2: `"{card}-evolution-in-hand"`

The `last_played_evo` field exposes which cards were just played as evo, used by the analyzer for on-field relabeling.

---

### Analysis Pipeline

#### VideoAnalyzer (`src/overlay/analyzer.py`)

Orchestrates frame-by-frame video processing:

1. **Detection**: Runs KataCR on frame
2. **Hand classification**: Runs HandClassifier on frame
3. **OCR**: Detects game timer, elixir count, multiplier icon
4. **Hand tracking**: Updates HandTracker with detections
5. **Post-processing rules**:
   - **Hero musketeer substitution**: If hero musketeer is in hand, any on-field musketeer is relabeled
   - **Evolution substitution**: Cards in `last_played_evo` get `-evolution` suffix on-field

6. **Output**: Per-frame FrameDict with all state

**Frame skip:** By default processes every 6th frame (configurable). At 60 FPS with skip=6, effective sampling is 10 FPS.

**Output format:**

```json
{
  "video_info": {
    "path": "...",
    "width": 1920,
    "height": 1080,
    "fps": 60.0,
    "duration_seconds": 120.5,
    "frame_skip": 6,
    "total_frames_processed": 1200
  },
  "frames": [
    {
      "timestamp_ms": 0,
      "frame_number": 0,
      "game_time_remaining": 180.0,
      "elixir_multiplier": 1.0,
      "game_phase": null,
      "player_elixir": 5.0,
      "opponent_elixir_estimated": null,
      "detections": [
        {
          "class_name": "hog-rider",
          "tile_x": 8,
          "tile_y": 15,
          "is_opponent": false,
          "is_on_field": true,
          "confidence": 0.94
        }
      ],
      "hand_cards": ["musketeer-in-hand", "fireball-in-hand", "zap-in-hand", "the-log-in-hand"],
      "player_towers": {},
      "opponent_towers": {}
    }
  ]
}
```

---

### Episode Generation

#### Episode Builder (`src/data/episode.py`)

Converts analysis JSON to training episodes by detecting card plays:

**Play detection:**
A play occurs when:
1. A card was in the tracked hand at the previous frame
2. A new on-field unit appears at `tile_y >= 17` (only lower half, excludes towers)
3. The card name matches (after normalization)

**Trajectory structure:**

```
Episode:
  [Frame 0]: state=frame_0, action=(card, tile), return=...
  [Frame 1]: state=frame_1, action=(card, tile), return=...
  [Frame 2]: state=frame_2, action=(card, tile), return=...
  ...
```

Each frame generates one `(state, action, return)` tuple. If no play occurred, action is None.

**Return-to-go (RTG):** Placeholder (all 0.0 for now). In future, could use win/loss or elixir advantage.

---

### Training Pipeline

#### Feature Encoder (`src/data/feature_encoder.py`)

Converts FrameDict to fixed-length 33-dim float32 vector:

```
[0]       player_elixir / 10
[1]       game_time / 180
[2]       multiplier / 3
[3-5]     player tower HP ratios (left, king, right)
[6-8]     opponent tower HP ratios
[9-28]    top 20 on-field detections (4 features each):
            - tile_x / 17
            - tile_y / 31
            - is_opponent (0 or 1)
            - card_id / VOCAB_SIZE
[29-32]   hand card IDs / VOCAB_SIZE (4 slots)
```

Missing detections are zero-padded. Hand cards are normalized (suffix removal) before vocabulary lookup.

#### Decision Transformer (`src/transformer/model.py`)

GPT-style causal transformer predicting card and placement:

**Architecture:**
- 4 layers, 4 attention heads, 128-dim embeddings
- Context length: 30 timesteps
- Input: (RTG, State, Action) triples interleaved into sequence
- Output: logits over card vocabulary (143) and tile positions (17×31)

**Loss:**
Weighted combination of card and position cross-entropy:
```
total_loss = 1.0 * card_loss + 1.0 * pos_loss
```

**Forward pass:**

```python
card_logits, pos_logits = model(
    states,           # (B, T, 33)
    actions_card,     # (B, T)
    actions_pos,      # (B, T)
    returns_to_go,    # (B, T, 1)
    timesteps,        # (B, T)
    attention_mask    # (B, T)
)
```

Predictions are masked (only training on non-padding tokens) and accuracy is computed per dimension.

#### Training Loop (`src/transformer/train.py`)

Orchestrates data loading, optimization, and checkpointing:

**Hyperparameters:**
- Learning rate: 1e-3 (configurable)
- Batch size: 64 (configurable)
- Max epochs: 600 (configurable)
- Weight decay: 1e-4
- Gradient clip: 1.0

**Validation metrics:**
- Card accuracy: % correct card predictions
- Position accuracy: % correct tile predictions
- Position top-5: % where correct tile in top 5 logits

**Checkpointing:**
- `best.pt`: Model with best validation card accuracy
- `epoch_N.pt`: Numbered checkpoint every 25 epochs (configurable)

---

## Data Flow

### 1. Video Analysis

```bash
python scripts/analyze_video.py data/replays/game_1.mp4 --device mps
```

Outputs: `output/analysis/game_1_analysis.json`

For all videos:
```bash
for v in data/replays/*.mp4; do
  python scripts/analyze_video.py "$v" --device mps
done
```

### 2. Episode Generation

```bash
python scripts/convert_to_episodes.py output/analysis --output output/pkl/episodes.pkl
```

Reads all JSON files from `output/analysis/`, builds Episode objects, saves combined pickle.

### 3. Model Training

```bash
python src/transformer/train.py \
  --episodes output/pkl/episodes.pkl \
  --epochs 600 \
  --device mps \
  --save-every 25
```

Trains for 600 epochs, saving checkpoints every 25 epochs. Monitors validation accuracy and saves best model.

---

## Directory Structure

```
TorchRoyale/
├── data/
│   ├── models/
│   │   ├── onfield/           # KataCR detection models
│   │   │   ├── detector1_v0.7.13.pt
│   │   │   └── detector2_v0.7.13.pt
│   │   └── hand_classifier/
│   │       └── hand_classifier.pt
│   └── replays/               # Input video files (.mp4)
│
├── output/
│   ├── analysis/              # Per-video JSON analysis
│   │   ├── game_1_analysis.json
│   │   └── game_2_analysis.json
│   └── pkl/
│       └── episodes.pkl       # Combined training episodes
│
├── logs/
│   ├── onfield/               # Debug images (on-field detections)
│   └── inhand/                # Debug images (hand detections)
│
├── src/
│   ├── constants/
│   │   ├── cards.py           # Vocabulary (143 cards), costs, mappings
│   │   └── game.py            # Grid dims (17x31), player side row
│   │
│   ├── detection/
│   │   ├── katacr.py          # On-field unit detection
│   │   ├── hand_classifier.py # Hand card classification
│   │   ├── hand_tracker.py    # Stateful hand state + evo tracking
│   │   └── result.py          # Detection and FrameDetections dataclasses
│   │
│   ├── overlay/
│   │   ├── analyzer.py        # Video frame processor
│   │   ├── inference_runner.py # Inference API (stub)
│   │   └── player.py          # Game state player (stub)
│   │
│   ├── ocr/
│   │   ├── detector.py        # OCR for timer, elixir, multiplier
│   │   └── regions.py         # UI region calibration
│   │
│   ├── data/
│   │   ├── feature_encoder.py # State → 33-dim vector
│   │   ├── dataset.py         # Data loading with train/val split
│   │   └── episode.py         # Episode building from analysis JSON
│   │
│   ├── transformer/
│   │   ├── model.py           # Decision Transformer architecture
│   │   └── train.py           # Training loop
│   │
│   ├── grid/
│   │   └── coordinate_mapper.py # Pixel ↔ tile conversions
│   │
│   └── types.py               # TypedDict definitions
│
├── scripts/
│   ├── analyze_video.py       # Video analysis script
│   ├── convert_to_episodes.py # Episode generation script
│   ├── test_katacr.py         # Debug: test KataCR on frame
│   └── test_hand_classifier.py # Debug: test hand classifier on frame
│
└── README.md
```

---

## Known Limitations

### Hog Rider Misclassification

KataCR models consistently misidentify hog riders as ice-spirit at ~0.94 confidence. This is a training limitation of the KataCR models — they were not trained on hog-rider as a distinct class.

**Impact:** Training episodes with hog riders will have incorrect unit labels.

**Workaround:** Manual annotation of affected frames or retraining KataCR models on custom Clash Royale dataset.

### Missing Tower State

Player and opponent tower HP is not tracked. `player_towers` and `opponent_towers` dicts are always empty in the output.

**Impact:** Feature encoder assigns all tower HP ratios to 1.0 (healthy). Model cannot distinguish remaining tower count or HP.

### Return-to-Go (RTG) Placeholder

RTG is always 0.0. Real implementation would use game outcome (win/loss) or elixir advantage metrics.

**Impact:** Model has no reward signal to distinguish winning vs losing plays.

---

## Running the Full Pipeline

### 1. Prepare Videos

Place replay video files in `data/replays/`:

```bash
ls data/replays/
# game_1.mp4
# game_2.mp4
# ...
```

### 2. Analyze Videos

```bash
for v in data/replays/*.mp4; do
  python scripts/analyze_video.py "$v" --device mps
done
```

Outputs per-frame JSON to `output/analysis/`.

### 3. Generate Episodes

```bash
python scripts/convert_to_episodes.py output/analysis --output output/pkl/episodes.pkl
```

Combines all analysis JSON into training episodes.

### 4. Train Model

```bash
python src/transformer/train.py \
  --episodes output/pkl/episodes.pkl \
  --epochs 600 \
  --batch-size 64 \
  --lr 1e-3 \
  --context-len 30 \
  --device mps \
  --save-every 25
```

Trains for 600 epochs with checkpoints every 25 epochs. Best model saved as `data/models/dt/best.pt`.

---

## Key Concepts

### Tile Coordinates

The game grid is 17 columns × 31 rows (GRID_COLS × GRID_ROWS in constants).

- `tile_x`: 0–16 (left to right)
- `tile_y`: 0–30 (top to bottom)
- Player side: `tile_y >= PLAYER_SIDE_MIN_ROW` (typically 15)
- Opponent side: `tile_y < 15`

### Card Normalization

Card names flow through multiple representations:

1. **KataCR output**: `"Ice-Spirit"`, `"Evolution"` (class names from model)
2. **Dataset format**: `"ice spirit"`, `"skeletons"`, `"evo fireball"` (spaces, plural, evo prefix)
3. **Canonical format**: `"ice-spirit"`, `"skeleton"`, `"fireball-evolution"` (dashes, singular, evo suffix)

The pipeline normalizes at multiple points:
- `katacr._base()`: Strips KataCR suffixes
- `hand_classifier._normalise()`: Converts dataset output to canonical
- `hand_tracker._base()`: Strips in-hand/evo suffixes for matching

### Feature Scaling

All features are normalized to [0, 1]:
- Elixir: `/10` (0–10 elixir)
- Timer: `/180` (0–180 seconds)
- Multiplier: `/3` (1–3x)
- Tower HP: 0.0 (destroyed) to 1.0 (healthy)
- Tile coordinates: `tile_x / 17`, `tile_y / 31`
- Card IDs: `/VOCAB_SIZE`

---

## Debugging

### Test KataCR Detector

```bash
python scripts/test_katacr.py data/replays/game_1.mp4 \
  --frame 1000 \
  --device mps
```

Outputs debug image with bounding boxes to `logs/onfield/game1_f1000.jpg`.

### Test Hand Classifier

```bash
python scripts/test_hand_classifier.py data/replays/game_1.mp4 \
  --frame 500 \
  --device mps
```

Outputs debug image with hand slot crops and classifications.

### Generate Validation Images

```bash
python scripts/test_katacr.py data/replays/game_1.mp4 \
  --frame 0 \
  --stride 120 \
  --count 20 \
  --device mps
```

Generates 20 images at 120-frame intervals to validate detection quality.

---

## Development Notes

### Adding New Cards

1. Add card to `src/constants/cards.py`:
   - `CARD_NAMES`: sorted list (maintains stable indices)
   - `ELIXIR_COSTS`: cost lookup

2. Update `HandClassifier` dataset if needed

3. No code changes required; feature encoder will use new vocabulary index

### Custom Device Support

All components support PyTorch device strings:
- `"auto"`: MPS > CUDA > CPU (intelligent fallback)
- `"mps"`: Apple Silicon GPU
- `"cuda"`: NVIDIA GPU
- `"cpu"`: CPU-only

Pass via `--device` flag to all scripts.

### Hyperparameter Tuning

Key parameters in `src/transformer/model.py` (DTConfig):
- `context_len`: Sequence length (default 30)
- `d_model`: Embedding dimension (default 128)
- `n_head`: Attention heads (default 4)
- `n_layer`: Transformer layers (default 4)
- `lr`: Learning rate (default 1e-3)
- `batch_size`: Batch size (default 64)

Modify via `--context-len`, `--lr`, `--batch-size` CLI flags or edit `DTConfig` defaults.

---

## References

- **KataCR**: https://github.com/yuanluya/KataCR (YOLOv8 Clash Royale models)
- **Decision Transformer**: https://arxiv.org/abs/2202.05566 (Offline RL via offline data)
- **Clash Royale**: https://supercell.com/en/games/clashroyale/ (Official game)


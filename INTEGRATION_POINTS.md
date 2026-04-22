# TorchRoyale DT Integration Points

## Quick Reference: Where TorchRoyale Decision Transformer Fits

### Status: ALREADY INTEGRATED

The Decision Transformer model is already fully integrated into the codebase. Below are the key integration points and how data flows through the system.

---

## 1. Entry Points for DT Inference

### For Video Analysis
```python
# File: src/overlay/inference_runner.py
from src.recommendation.strategy import DTStrategy
from src.video.video_analyzer import VideoAnalyzer

# Run full video analysis
analyzer = VideoAnalyzer(frame_skip=6)
result = analyzer.analyze_video("game.mp4")

# Then run DT inference on results
runner = InferenceRunner(
    video_path="game.mp4",
    checkpoint_path="data/models/dt_checkpoint.pt",
    output_jsonl="recommendations.jsonl"
)
recommendations = runner.run()
```

### For Real-Time Processing
```python
# File: pipeline_live.py
from pipeline_live import LivePipeline
from src.recommendation.strategy import DTStrategy

pipeline = LivePipeline(target_fps=30.0)
strategy = DTStrategy(checkpoint_path="data/models/dt_checkpoint.pt")
strategy.reset_game()

while True:
    ret, frame = cap.read()
    state = pipeline.process_frame(frame)
    
    if state and state['elixir'] > 0:
        rec = strategy.recommend(state)
        if rec:
            card, tile_x, tile_y = rec
            # Use recommendation
```

---

## 2. Data Flow to DT Model

### Path A: Raw Detection → Feature Encoding → DT

```
FrameState Dict
    ├─ timestamp_ms: int
    ├─ player_elixir: int (0-10)
    ├─ opponent_elixir_estimated: float
    ├─ game_phase: str ("single"/"double"/"triple"/"sudden_death")
    ├─ game_time_remaining: Optional[int] (seconds)
    ├─ detections: List[Detection]
    │   ├─ class_name: str
    │   ├─ tile_x, tile_y: int (0-17, 0-31)
    │   └─ is_opponent: bool
    ├─ player_towers: Dict[str, Dict]
    ├─ opponent_towers: Dict[str, Dict]
    └─ hand_cards: List[str]
        ↓
        feature_encoder.encode(state_dict)
        ↓
    Feature Vector (97 dimensions)
        - [0]: player_elixir / 10
        - [1]: opponent_elixir / 10
        - [2:6]: game_phase one-hot
        - [6]: time_remaining / 300
        - [7:10]: player tower ratios
        - [10:13]: opponent tower ratios
        - [13:93]: 20 detections (4 features each)
        - [93:97]: hand cards
        ↓
        DTStrategy.recommend(state_dict)
        ↓
    Recommendation: (card_name, tile_x, tile_y)
```

### Path B: Direct DTInference Access

```python
# For advanced usage
from src.transformer.inference import DTInference
from src.data.feature_encoder import encode

# Load model directly
inference = DTInference(
    checkpoint_path="data/models/dt_checkpoint.pt",
    device="cpu",
    target_return=100.0,
    temperature=1.0
)

# Feed feature vector
feature_vec = encode(frame_state)
card_idx, pos_flat = inference.predict(frame_state)

# Update context window for next prediction
inference.update_action(card_idx, pos_flat)
```

---

## 3. Model Configuration

### Location: configs/inference.yaml
```yaml
checkpoint_path: "data/models/dt_checkpoint.pt"
target_return: 100.0
device: "cpu"  # or "cuda" for GPU
temperature: 1.0
fallback_tile_x: 9
fallback_tile_y: 24
```

### Override at Runtime
```python
strategy = DTStrategy(
    checkpoint_path="path/to/custom_model.pt",
    target_return=150.0,  # Override: aim for higher win margin
    device="cuda",
    temperature=0.8  # More deterministic (vs 1.0 = sample)
)
```

---

## 4. Context Window Management

### Automatic Context Updates
```python
# DTInference maintains rolling context window

# Reset for new game
strategy.reset_game()

# Each recommendation updates context
for frame_state in game_frames:
    card_idx, pos_flat = strategy._inference.predict(frame_state)
    strategy._inference.update_action(card_idx, pos_flat)
    
    # Context resets every 20 steps to prevent feedback accumulation
    # This is automatic within DTInference
```

### Context Window Contents
```python
self._states: List[np.ndarray]    # History of feature vectors
self._cards: List[int]             # Cards played (0-3)
self._positions: List[int]         # Tile positions (0-575)
self._rtgs: List[float]            # Return-to-go values
self._current_rtg: float           # Target return for conditioning
```

---

## 5. Feature Encoding Details

### FrameState to Vector Conversion

```python
# src/data/feature_encoder.py
from src.data.feature_encoder import encode, FEATURE_DIM

state = frame_state.to_dict()  # FrameState → dict
feature_vector = encode(state)  # 97-dim vector

# Feature dimensions:
FEATURE_DIM = 97
# Breakdown:
# - Elixir: 2 (player, opponent)
# - Phase: 4 (one-hot: single/double/triple/sudden_death)
# - Time: 1
# - Player towers: 3 (left/king/right HP ratios)
# - Opponent towers: 3
# - Detections: 80 (20 detections × 4 features)
#   - class_id/vocab_size
#   - tile_x/17
#   - tile_y/31
#   - is_opponent
# - Hand cards: 4 (card IDs / vocab_size)
```

---

## 6. Game State Data Sources

### What Components Feed Into Feature Encoding

| Component | Output | Integration |
|-----------|--------|-------------|
| **Roboflow** | detections (cards, towers) | detection_test.py |
| **EasyOCR** | elixir (0-10), timer (seconds), multiplier (1/2/3) | ocr/digit_detector.py |
| **GamePhaseTracker** | game_phase enum, elixir_multiplier | game_state/game_phase.py |
| **OpponentElixirTracker** | opponent_elixir (0-10 float) | recommendation/elixir_manager.py |
| **TowerHealthDetector** | tower HP and max HP via OCR | game_state/health_detector.py |
| **CoordinateMapper** | pixel → tile grid conversion | grid/coordinate_mapper.py |

### Example: Elixir Tracking Flow

```
EasyOCR detect_elixir()
    ↓ (returns 0-10 int)
PlayerElixirTracker.update()
    ↓ (applies smoothing, tracking)
player_elixir (in FrameState)
    ↓ (added to feature vector)
Feature[0] = player_elixir / 10
    ↓ (normalized to 0-1)
DTInference.predict()
```

---

## 7. Output Formats

### Recommendation Output
```python
recommendation = strategy.recommend(frame_state)

# Returns:
if recommendation:
    card_name: str       # "hog-rider", "ice-golem", etc.
    tile_x: int          # 0-17 (horizontal)
    tile_y: int          # 0-31 (vertical, 17-31 is friendly side)
else:
    None                 # No affordable card
```

### JSONL Output (InferenceRunner)
```json
{"timestamp_ms": 1000, "player_elixir": 8, "has_recommendation": true, 
 "card": "hog-rider", "tile_x": 9, "tile_y": 24, "elixir_required": 4}

{"timestamp_ms": 2000, "player_elixir": 4, "has_recommendation": false}

{"timestamp_ms": 3000, "player_elixir": 6, "has_recommendation": true, 
 "card": "ice-spirit", "tile_x": 10, "tile_y": 20, "elixir_required": 1}
```

---

## 8. Fallback Behavior

### When DT Model Not Available
```python
strategy = DTStrategy(checkpoint_path="nonexistent_model.pt")

if not strategy.is_ready:
    # Falls back to heuristic
    # Plays most expensive affordable card at default position
    # (default: tile_x=9, tile_y=24)
    recommendation = strategy.recommend(frame_state)
```

### Placement Validation
```python
# DTStrategy automatically validates placements
card_name = "ice-spirit"
tile_x, tile_y = rec  # From DT model

# Check if valid for this card
if not validator.is_valid_placement(card_name, tile_x, tile_y):
    # Find closest valid tile
    valid_tiles = validator.get_valid_tiles(card_name)
    closest = min(valid_tiles, 
                  key=lambda t: abs(t[0]-tile_x) + abs(t[1]-tile_y))
    tile_x, tile_y = closest
```

---

## 9. Real-Time vs Batch Processing

### Real-Time (LivePipeline)
```python
# Use for live gameplay or stream processing
from pipeline_live import LivePipeline
from src.recommendation.strategy import DTStrategy

pipeline = LivePipeline(target_fps=30.0)
strategy = DTStrategy()
strategy.reset_game()

while game_running:
    frame = capture_frame()
    state = pipeline.process_frame(frame)
    
    # Recommendation available immediately
    rec = strategy.recommend(state)
    
    # Act on recommendation
    if rec:
        execute_action(rec)
```

**Performance:**
- 15-30 FPS achievable
- OCR cached every 3 frames (10 Hz)
- Detection cached every 2 frames (15 Hz)
- DT inference every frame (30 Hz possible)

### Batch (VideoAnalyzer)
```python
# Use for game analysis and dataset building
from src.video.video_analyzer import VideoAnalyzer
from src.overlay.inference_runner import InferenceRunner

# 1. Analyze video
analyzer = VideoAnalyzer(frame_skip=6)
result = analyzer.analyze_video("game.mp4")

# 2. Generate recommendations
runner = InferenceRunner(
    video_path="game.mp4",
    checkpoint_path="models/dt_checkpoint.pt",
    output_jsonl="recommendations.jsonl"
)
runner.run()
```

**Performance:**
- ~5 FPS with frame_skip=6
- All frames processed (no caching)
- Full game state JSON output

---

## 10. Coordinate System for DT Output

### Tile Grid Layout (18x32)
```
        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17  (tile_x)
     ┌────────────────────────────────────────────────────────┐
  0  │                  OPPONENT SIDE                           │
  .  │                  (rows 0-14)                            │
 14  │──────────────────────────────────────────────────────── │
 15  │         RIVER (bridges at x=3,4,13,14)                 │
 16  │──────────────────────────────────────────────────────── │
 17  │                   YOUR SIDE                              │
  .  │                 (rows 17-31)                            │
 31  └────────────────────────────────────────────────────────┘
    (tile_y)

DT Output: (tile_x=0-17, tile_y=0-31)
Friendly placement: tile_y must be in [17-31]
Opponent detection: tile_y in [0-14]
```

### Pixel ↔ Tile Conversion
```python
from src.grid.coordinate_mapper import CoordinateMapper

mapper = CoordinateMapper()

# DT outputs tile coordinates
tile_x, tile_y = 9, 24

# Convert to pixel coordinates for display/action
pixel_x, pixel_y = mapper.tile_to_pixel(tile_x, tile_y, center=True)

# Reverse conversion
detected_pixel_x, detected_pixel_y = 100, 500  # from Roboflow
tile_x, tile_y = mapper.pixel_to_tile(detected_pixel_x, detected_pixel_y)
```

---

## 11. Dependencies for DT Inference

### Minimum Required
```
torch >= 2.0
numpy >= 1.24
```

### For Full System
```
pytorch >= 2.0
opencv-python >= 4.8
easyocr >= 1.6
roboflow >= 1.1
supervision >= 0.19
numpy >= 1.24
scikit-learn (for fallback MLStrategy)
joblib
```

---

## 12. Testing DT Integration

### Minimal Test
```python
from src.recommendation.strategy import DTStrategy

# Create strategy
strategy = DTStrategy(
    checkpoint_path="data/models/dt_checkpoint.pt",
    device="cpu"
)

# Test state
test_state = {
    "player_elixir": 8,
    "opponent_elixir_estimated": 5.0,
    "game_phase": "single",
    "game_time_remaining": 120,
    "hand_cards": ["hog-rider", "ice-spirit", "cannon", "the-log"],
    "player_towers": {
        "player_left": {"health_percent": 100.0},
        "player_king": {"health_percent": 85.0},
        "player_right": {"health_percent": 100.0}
    },
    "opponent_towers": {
        "opponent_left": {"health_percent": 90.0},
        "opponent_king": {"health_percent": 75.0},
        "opponent_right": {"health_percent": 90.0}
    },
    "detections": [],
}

# Get recommendation
rec = strategy.recommend(test_state)
if rec:
    card, tile_x, tile_y = rec
    print(f"Play {card} at ({tile_x}, {tile_y})")
else:
    print("No recommendation (insufficient elixir or no valid placement)")
```

---

## 13. Advanced: Custom DTInference Usage

### Accessing Raw Model Output
```python
from src.transformer.inference import DTInference
from src.data.feature_encoder import encode

inference = DTInference(
    checkpoint_path="models/dt_checkpoint.pt",
    device="cpu",
    target_return=100.0,
    temperature=1.0
)

# Reset for new game
inference.reset()

# Get raw predictions
for state in game_states:
    card_idx, pos_flat = inference.predict(state)
    
    # card_idx: 0-3 (which hand slot)
    # pos_flat: 0-575 (tile position)
    #   - pos_flat = tile_y * 18 + tile_x
    
    tile_x = pos_flat % 18
    tile_y = pos_flat // 18
    
    # Update context for next prediction
    inference.update_action(card_idx, pos_flat)
```

### Context Reset Interval
```python
# DTInference resets context every 20 steps by default
# This prevents autoregressive feedback from accumulating

# You can check step count:
steps = inference._steps_since_reset

# Manual reset if needed:
inference._reset_context()
```

---

## 14. Integration Checklist

- [x] DTStrategy class exists and is functional
- [x] Feature encoding (97-dim) works correctly
- [x] FrameState data structure is complete
- [x] Real-time pipeline (LivePipeline) supports DT inference
- [x] Video analysis pipeline (VideoAnalyzer) supports DT
- [x] JSONL output format for recommendations
- [x] Coordinate mapping (pixel ↔ tile)
- [x] Placement validation
- [x] Fallback strategies when model unavailable
- [x] Config system for model parameters

### For Live Gameplay Automation (TODO)
- [ ] ADB device controller
- [ ] Screen capture integration
- [ ] Input injection (tap coordinates)
- [ ] PyQt GUI (optional, for visualization)
- [ ] Event loop orchestrating pipeline + strategy

---

## Quick Links

- Architecture Details: `/ARCHITECTURE.md`
- Real-Time Pipeline: `/pipeline_live.py`
- Video Analysis: `/src/video/video_analyzer.py`
- DT Strategy: `/src/recommendation/strategy.py`
- DT Inference: `/src/transformer/inference.py`
- Feature Encoder: `/src/data/feature_encoder.py`


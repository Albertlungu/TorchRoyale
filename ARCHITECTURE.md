# TorchRoyale Codebase Architecture Map

## Overview
TorchRoyale is a Decision Transformer-based bot for Clash Royale that processes video frames, detects game state, and recommends card placements. The codebase is organized into modular components with clear data flow:

```
Video Input → Frame Processing → Detection → State Analysis → Recommendation Model → Output
```

---

## 1. Project Structure

```
/Users/albertlungu/Local/GitHub/TorchRoyale/
├── src/                          # Main source code
│   ├── capture/                  # Video/screen capture (currently empty)
│   ├── classifier/               # Card classification
│   │   └── classifier.py
│   ├── constants/                # Game data and config
│   │   ├── card_costs.py
│   │   ├── card_types.py
│   │   ├── game_constants.py     # Elixir rates, phases, tower HP
│   │   └── ui_regions.py         # UI element pixel regions
│   ├── data/                      # Data processing and feature encoding
│   │   ├── dt_dataset.py
│   │   ├── episode_builder.py
│   │   ├── feature_encoder.py    # State → fixed-size feature vector
│   │   ├── label_extractor.py
│   │   └── outcome_detector.py
│   ├── detection/                # Object detection (Roboflow)
│   │   ├── board_analyzer.py
│   │   ├── card_detector.py
│   │   └── model_loader.py
│   ├── game_state/               # Game state tracking
│   │   ├── game_phase.py         # Single/Double/Triple/Sudden Death phases
│   │   ├── health_detector.py    # Tower health OCR
│   ├── grid/                      # Coordinate transformation
│   │   ├── coordinate_mapper.py  # Pixel ↔ Tile grid mapping (18x32)
│   │   └── validity_masks.py     # Card placement validity
│   ├── ocr/                       # Optical Character Recognition
│   │   ├── digit_detector.py     # EasyOCR for elixir/timer detection
│   │   └── vision_detector.py
│   ├── overlay/                   # Inference and replay runners
│   │   ├── inference_runner.py   # Process video → JSONL recommendations
│   │   ├── replay_runner.py
│   │   └── video_player.py
│   ├── recommendation/            # Decision making
│   │   ├── elixir_manager.py     # Player/opponent elixir tracking
│   │   ├── model_trainer.py
│   │   └── strategy.py           # MLStrategy and DTStrategy
│   ├── transformer/              # Decision Transformer model
│   │   ├── config.py             # DTConfig for model architecture
│   │   ├── inference.py          # DTInference wrapper with context window
│   │   ├── model.py              # Decision Transformer implementation
│   │   └── train.py
│   ├── ui/                        # GUI components (mostly empty)
│   │   ├── card_display.py
│   │   ├── grid_overlay.py
│   │   └── main_window.py
│   ├── utils/                     # Utilities
│   │   ├── config.py
│   │   ├── inference_config.py   # InferenceConfig for DT settings
│   │   └── logger.py
│   └── video/                     # Video processing
│       ├── video_analyzer.py      # Main orchestration pipeline
│       └── video_processor.py     # Frame extraction with skip intervals
│
├── pipeline_live.py               # Real-time processing pipeline
├── pipeline_live_test.py
├── pipeline_test.py
├── detection_test.py              # DetectionPipeline for Roboflow
├── tests/                         # Unit tests
├── data/                          # Models and data
├── configs/                       # Configuration files
└── docs/                          # Documentation
```

---

## 2. Frontend/GUI Architecture

### Current Status
The UI components in `/src/ui/` are **mostly empty stubs**:
- `main_window.py` (empty)
- `card_display.py` (empty)
- `grid_overlay.py` (empty)

**No PyQt6 implementation currently exists.**

### Integration Points for TorchRoyale DT
The system is designed for **backend-only operation** (processing videos → generating recommendations). To add GUI:

1. **Display Components Needed:**
   - Real-time frame viewer
   - Game state overlay (elixir, phase, towers)
   - Detection visualization (bounding boxes)
   - Recommendation overlay (suggested card + placement tile)

2. **Data Sources for UI:**
   - `LivePipeline.process_frame()` returns game state dict with all display info
   - `DetectionPipeline.process_image_array()` provides detections and annotated frames

---

## 3. ADB Integration

### Current Status
**No ADB integration exists in this codebase.** The system is video-only (post-match analysis and live video input).

### Required for Live Play:
Would need to add:
- Screen capture via ADB (Android emulator/phone)
- Input injection (tap coordinates)
- Connection management

**Recommended approach:**
```python
class ADBController:
    - connect_device(device_id)
    - capture_frame() → np.ndarray
    - tap(tile_x, tile_y)
    - swipe(x1, y1, x2, y2)
    - screen_size() → (width, height)
```

---

## 4. Game State Management

### FrameState Data Structure
(Location: `/src/video/video_analyzer.py`)

```python
@dataclass
class FrameState:
    # Timing
    timestamp_ms: int
    frame_number: int
    
    # Game phase
    game_phase: str              # "single", "double", "triple", "sudden_death"
    elixir_multiplier: int       # 1, 2, or 3
    
    # Elixir
    player_elixir: int
    opponent_elixir_estimated: float
    
    # Timer
    game_time_remaining: Optional[int]  # seconds
    
    # Detections
    detections: List[Detection]
    
    # Tower health
    player_towers: Dict[str, Dict]
    opponent_towers: Dict[str, Dict]
    
    # Cards in hand
    hand_cards: List[str]
```

### State Update Pipeline

```
Frame (cv2.Mat)
    ↓
VideoAnalyzer._process_frame()
    ├─ Roboflow Detection → detections list
    ├─ EasyOCR Detection
    │   ├─ detect_elixir() → player_elixir
    │   ├─ detect_timer() → game_time_remaining
    │   └─ detect_multiplier_icon() → elixir_multiplier (1/2/3)
    ├─ GamePhaseTracker.update() → game_phase
    ├─ OpponentElixirTracker.update() → opponent_elixir
    ├─ TowerHealthDetector.detect_all_towers() → tower health
    └─ Returns FrameState
```

### Key State Components

#### 4.1 Game Phase Tracker
**File:** `/src/game_state/game_phase.py`
**Class:** `GamePhaseTracker`

State transitions:
```
SINGLE_ELIXIR (3:00-2:00)
    ↓ (timer ≤ 60s or x2 detected)
DOUBLE_ELIXIR (2:00-0:00)
    ├─ (timer = 0:00, towers tied)
    │   ↓
    │  SUDDEN_DEATH
    │      ↓ (x3 detected or last 60s of overtime)
    │   TRIPLE_ELIXIR
    │      ↓
    │   GAME_OVER
    │
    └─ (timer = 0:00, not tied)
        ↓
        GAME_OVER
```

**Input signals:**
- `multiplier_detected` (1/2/3) from OCR
- `timer_seconds` from OCR
- `towers_tied` from detection logic

#### 4.2 Opponent Elixir Tracking
**File:** `/src/recommendation/elixir_manager.py`
**Class:** `OpponentElixirTracker`

Calculation:
```
opponent_elixir = 5 (start)
  + time_elapsed * regen_rate(phase) / seconds_per_elixir
  - sum(card_costs_played)
```

Regeneration rates:
- Single elixir: 2.8 seconds per 1 elixir
- Double elixir: 1.4 seconds per 1 elixir
- Triple elixir: 0.933 seconds per 1 elixir

#### 4.3 Tower Health Detection
**File:** `/src/game_state/health_detector.py`
**Class:** `TowerHealthDetector`

- Detects tower HP via OCR (EasyOCR on health bar numbers)
- Detects tower level (once) from visual features
- Stores last known HP for princess towers (OCR often fails)
- Max HP varies by tower level (level 1-16)

---

## 5. Detection Pipeline

### Roboflow Integration
**File:** `/detection_test.py`
**Class:** `DetectionPipeline`

```python
DetectionPipeline:
    - __init__(model_id="torchroyale/4")
    - process_image_array(frame: np.ndarray)
        ├─ Roboflow inference → raw detections
        ├─ Coordinate mapping (pixel → tile)
        ├─ Classification (on-field vs in-hand, friendly vs opponent)
        └─ Returns annotated image + detection list
```

### Detection Objects
**Dataclass:** `Detection`

```python
Detection:
    class_name: str           # e.g., "hog-rider", "bandit", "king-tower"
    confidence: float         # Roboflow confidence
    pixel_x, pixel_y: int     # From Roboflow
    pixel_width, pixel_height: int
    tile_x, tile_y: int       # Mapped to 18x32 grid
    is_opponent: bool
    is_on_field: bool         # vs in-hand cards
```

### Coordinate Mapping
**File:** `/src/grid/coordinate_mapper.py`
**Class:** `CoordinateMapper`

Grid layout:
```
          0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 (x)
     ┌─────────────────────────────────────────────────────────┐
  0  │                    OPPONENT SIDE                         │
  .  │                     (rows 0-14)                          │
 14  │─────────────────────────────────────────────────────────│
 15  │                      RIVER                               │ (bridges at x=3,4,13,14)
 16  │─────────────────────────────────────────────────────────│
 17  │                     YOUR SIDE                            │
  .  │                    (rows 17-31)                          │
 31  │─────────────────────────────────────────────────────────│
     └─────────────────────────────────────────────────────────┘
```

Conversion:
```
pixel_x, pixel_y (from Roboflow)
    ↓ subtract arena offset
arena_x, arena_y
    ↓ divide by tile_size
tile_x = 0-17, tile_y = 0-31
```

Calibration (default for 1170x2532):
- Arena origin: (28, 327) pixels
- Tile size: 62×50 pixels
- Arena: 18 tiles wide, 32 tiles tall

### Placement Validity
**File:** `/src/grid/validity_masks.py`

Card placement rules:
- Ranged units (archer): cannot place on river, must be on friendly side
- Melee units (knight): any friendly tile
- Deployables: restricted placement zones
- On-field vs hand detection

---

## 6. Feature Encoding

### FrameState → Feature Vector
**File:** `/src/data/feature_encoder.py`
**Function:** `encode(state: Dict) → np.ndarray`

Feature vector (~97 dimensions):

```
[0]      player_elixir / 10
[1]      opponent_elixir / 10
[2:6]    game_phase one-hot (single=0, double=1, triple=2, sudden_death=3)
[6]      time_remaining / 300
[7:10]   player tower HP ratios (left, king, right)
[10:13]  opponent tower HP ratios (left, king, right)
[13:93]  board state: up to 20 detections
         each: (class_id/vocab_size, tile_x/17, tile_y/31, is_opponent)
[93:97]  hand cards: 4 slots with card IDs / vocab_size
```

**Card Vocabulary:**
- Built from ELIXIR_COSTS keys
- Index 0 reserved for padding/unknown
- Used in feature normalization

---

## 7. Recommendation Strategy

### Two Implementations Available

#### 7.1 MLStrategy (Traditional)
**File:** `/src/recommendation/strategy.py`
**Class:** `MLStrategy`

Two-stage Random Forest:
1. **Stage 1:** Predict which hand card (0-3)
2. **Stage 2:** Predict placement tile (0-575)

Inference:
```python
strategy = MLStrategy()
if strategy.is_ready:  # models loaded
    card, tile_x, tile_y = strategy.recommend(frame_state)
else:
    # Fallback: play most expensive affordable card at center
```

#### 7.2 DTStrategy (Decision Transformer)
**File:** `/src/recommendation/strategy.py`
**Class:** `DTStrategy`

Transformer-based with context window:
```python
strategy = DTStrategy(
    checkpoint_path="models/dt_checkpoint.pt",
    target_return=100.0,  # target win margin
    device="cpu",
    temperature=1.0
)

strategy.reset_game()  # new game
card, tile_x, tile_y = strategy.recommend(frame_state)
strategy._inference.update_action(card_idx, pos_flat)  # update context
```

**Context Window Management:**
- Maintains history of past states, cards, positions
- Resets periodically (every 20 steps) to prevent feedback accumulation
- RTG (return-to-go) tracking for reinforcement signal

### Model Configuration
**File:** `/src/utils/inference_config.py`
**Class:** `InferenceConfig`

```yaml
checkpoint_path: "data/models/dt_checkpoint.pt"
target_return: 100.0
device: "cpu"
temperature: 1.0
fallback_tile_x: 9
fallback_tile_y: 24
```

---

## 8. Real-Time Processing Pipeline

### LivePipeline
**File:** `/pipeline_live.py`
**Class:** `LivePipeline`

Optimized for speed:
```python
pipeline = LivePipeline(target_fps=30.0)

while True:
    ret, frame = cap.read()
    state = pipeline.process_frame(frame)
    # state contains: elixir, opponent_elixir, timer, game_phase,
    #                 tower_health, detections, processing_time
```

**Caching Strategy:**
- OCR every 3 frames (10 Hz from 30 FPS target)
- Detection every 2 frames (15 Hz)
- Tower health every 15 frames (2 Hz)
- Tower level detection once

**Performance:**
- Target: 30 FPS
- Typical: 15-30 FPS with full pipeline

---

## 9. Video Analysis Pipeline

### VideoAnalyzer
**File:** `/src/video/video_analyzer.py`
**Class:** `VideoAnalyzer`

Complete end-to-end processing:
```python
analyzer = VideoAnalyzer(
    frame_skip=6,              # process every 6th frame (~5 FPS from 30 FPS)
    save_annotated_frames=False,
    output_dir="output"
)

result = analyzer.analyze_video("game.mp4")
# Returns:
# {
#   "video_info": {...},
#   "frames": [FrameState.to_dict(), ...],
#   "summary": {...}
# }
```

**Outputs:**
1. JSON file: `{video_stem}_analysis.json`
   - Frame-by-frame game state
   - Detections
   - Tower health
   - Elixir tracking

2. Optionally: Annotated frames as images

### InferenceRunner
**File:** `/src/overlay/inference_runner.py`
**Class:** `InferenceRunner`

Processes video + generates recommendations:
```python
runner = InferenceRunner(
    video_path="game.mp4",
    checkpoint_path="models/dt.pt",
    output_jsonl="recommendations.jsonl",
    frame_skip=6
)

runner.run()
# Writes JSONL with per-frame recommendations
```

JSONL format:
```json
{"timestamp_ms": 1000, "player_elixir": 8, "has_recommendation": true,
 "card": "hog-rider", "tile_x": 9, "tile_y": 24, "elixir_required": 4}
{"timestamp_ms": 2000, "player_elixir": 6, "has_recommendation": false}
```

---

## 10. Data Flow Diagrams

### Video → State Analysis → Recommendation

```
Video File (.mp4)
    ↓ VideoProcessor.frames()
Frame (np.ndarray, frame_num, timestamp_ms)
    ├─ Detection Pipeline
    │   ├─ Roboflow model inference → detections
    │   └─ Coordinate mapper: pixel → tile grid
    │
    ├─ OCR Pipeline (EasyOCR)
    │   ├─ detect_elixir() → player_elixir (0-10)
    │   ├─ detect_timer() → game_time_remaining (seconds)
    │   └─ detect_multiplier_icon() → multiplier (1/2/3)
    │
    ├─ State Trackers
    │   ├─ GamePhaseTracker.update() → game_phase
    │   ├─ OpponentElixirTracker.update() → opponent_elixir
    │   └─ TowerHealthDetector.detect_all_towers() → tower_health
    │
    └─ FrameState (complete game state for this frame)
        ↓
        Feature Encoder
        ↓
        Feature Vector (97 dims)
        ↓
        ┌─────────────────────────┐
        │  MLStrategy or DTStrategy │
        └─────────────────────────┘
        ↓
        Recommendation: (card_name, tile_x, tile_y)
```

### Real-Time Loop

```
Live Video Capture (camera/emulator)
    ↓ every frame
LivePipeline.process_frame()
    ├─ [Every 3 frames] OCR (elixir, timer, multiplier)
    ├─ [Every 2 frames] Roboflow detection
    ├─ [Every 15 frames] Tower health detection
    ├─ [Every frame] Game phase update
    ├─ [Every frame] Opponent elixir update
    │
    └─ Returns: game_state_dict
        {
            "elixir": int,
            "opponent_elixir": float,
            "timer": Optional[int],
            "game_phase": GamePhase,
            "tower_health": Dict,
            "detections": List[Detection] or None,
            "processing_time": float
        }
```

---

## 11. Integration Points for TorchRoyale DT

### Where to Insert DT Model

**Current Integration (Video Analysis):**
```python
# In VideoAnalyzer._process_frame() or pipeline_live.py
frame_state = {...}  # FrameState or dict

# Feature encoding happens automatically in strategy
recommendation = strategy.recommend(frame_state)
# Returns: (card_name, tile_x, tile_y) or None
```

**For Live Play Integration:**
```python
class GameBot:
    def __init__(self):
        self.adb = ADBController()
        self.pipeline = LivePipeline()
        self.strategy = DTStrategy()
        self.strategy.reset_game()
    
    def run_game(self):
        while not game_over:
            frame = self.adb.capture_frame()
            state = self.pipeline.process_frame(frame)
            
            if state and state['elixir'] > 0:
                rec = self.strategy.recommend(state)
                if rec:
                    card, tile_x, tile_y = rec
                    # Convert to pixel coords
                    px, py = self.mapper.tile_to_pixel(tile_x, tile_y)
                    self.adb.tap(px, py)
```

### Data Sources Available

| Component | Output | Type | Update Rate |
|-----------|--------|------|------------|
| DetectionPipeline | detections, annotated_frame | List[Detection], np.ndarray | 15 Hz |
| DigitDetector | elixir, timer, multiplier | int, int, int | 10 Hz |
| GamePhaseTracker | current_phase, elixir_multiplier | GamePhase, int | Real-time |
| OpponentElixirTracker | opponent_elixir | float | Real-time |
| TowerHealthDetector | tower_health | Dict[str, TowerHealthResult] | 2 Hz |
| DTStrategy | (card_idx, pos_flat) | (int, int) | Real-time |

### Required Dependencies

**For TorchRoyale DT Inference:**
- ✓ pytorch / torch
- ✓ numpy
- ✓ Feature encoder (already in codebase)
- ✓ Coordinate mapper (already in codebase)
- ✓ Placement validator (already in codebase)

**For Video Analysis:**
- ✓ opencv-cv2
- ✓ easyocr
- ✓ roboflow/inference SDK

**For Live Play (TODO):**
- adb-shell or similar
- Screen capture library

---

## 12. Key Files Reference

### Core Processing
| File | Purpose |
|------|---------|
| `pipeline_live.py` | Real-time frame processing |
| `src/video/video_analyzer.py` | Full video analysis orchestration |
| `src/overlay/inference_runner.py` | Video → recommendations JSONL |
| `detection_test.py` | Roboflow detection pipeline |

### State & Game Logic
| File | Purpose |
|------|---------|
| `src/game_state/game_phase.py` | Elixir phase state machine |
| `src/recommendation/elixir_manager.py` | Opponent elixir calculation |
| `src/game_state/health_detector.py` | Tower health OCR |

### Models & Inference
| File | Purpose |
|------|---------|
| `src/recommendation/strategy.py` | MLStrategy + DTStrategy |
| `src/transformer/inference.py` | DTInference wrapper |
| `src/transformer/model.py` | Decision Transformer architecture |

### Data Processing
| File | Purpose |
|------|---------|
| `src/data/feature_encoder.py` | FrameState → feature vector |
| `src/grid/coordinate_mapper.py` | Pixel ↔ tile grid mapping |
| `src/grid/validity_masks.py` | Card placement rules |
| `src/constants/game_constants.py` | Game mechanics data |

### OCR & Detection
| File | Purpose |
|------|---------|
| `src/ocr/digit_detector.py` | EasyOCR wrapper |
| `src/constants/ui_regions.py` | UI element regions |

---

## 13. Dependencies & Configuration

### Python Packages
```
pytorch >= 2.0
opencv-python >= 4.8
easyocr >= 1.6
roboflow >= 1.1
supervision >= 0.19
numpy >= 1.24
torch >= 2.0
joblib >= 1.3
```

### Config Files
```
configs/inference.yaml          # DTStrategy settings
configs/training.yaml           # (if training)
```

### Models
```
data/models/
  ├── stage1_card.pkl           # MLStrategy stage 1 (Random Forest)
  ├── stage2_pos.pkl            # MLStrategy stage 2 (Random Forest)
  └── dt_checkpoint.pt          # DTStrategy checkpoint
```

---

## 14. Summary: System Architecture

**Purpose:** Analyze Clash Royale gameplay videos and recommend card placements using Decision Transformer

**Architecture Pattern:** Pipeline with pluggable components

**Data Flow:**
1. **Input**: Video file or live frame stream
2. **Processing**: Multi-component analysis (detection, OCR, state tracking)
3. **State**: FrameState dict with all detected information
4. **Encoding**: Fixed-size feature vector for model input
5. **Inference**: DTStrategy.recommend() → (card, tile_x, tile_y)
6. **Output**: JSONL with recommendations or live gameplay

**Key Features:**
- ✓ Modular design (components are independent)
- ✓ Resolution-independent (UI regions use ratios)
- ✓ Real-time capable (15-30 FPS)
- ✓ Fallback strategies (when models unavailable)
- ✓ Caching for expensive operations

**Current Limitations:**
- ✗ No GUI (UI files are empty stubs)
- ✗ No ADB integration (video-only)
- ✗ No live device control

**Integration Path for TorchRoyale DT:**
The DT model is already integrated. To deploy live:
1. Add ADB device controller
2. Add PyQt GUI (optional)
3. Wire LivePipeline + DTStrategy into event loop
4. Handle device resolution calibration


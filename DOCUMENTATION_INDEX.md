# TorchRoyale Codebase Documentation Index

## Generated Documentation Files

This directory now contains comprehensive architecture and integration documentation for the TorchRoyale project.

### 1. ARCHITECTURE.md (22 KB)
**Comprehensive technical deep-dive**

The most detailed document covering all 14 aspects of the system architecture:

- **Section 1:** Project Structure (directory tree with descriptions)
- **Section 2:** Frontend/GUI Architecture (current status: empty stubs)
- **Section 3:** ADB Integration (current status: not implemented)
- **Section 4:** Game State Management (FrameState, tracking, phase states)
- **Section 5:** Detection Pipeline (Roboflow, Detection objects, coordinate mapping)
- **Section 6:** Feature Encoding (97-dim feature vectors)
- **Section 7:** Recommendation Strategy (MLStrategy vs DTStrategy)
- **Section 8:** Real-Time Processing Pipeline (LivePipeline caching strategy)
- **Section 9:** Video Analysis Pipeline (VideoAnalyzer, InferenceRunner)
- **Section 10:** Data Flow Diagrams (ASCII diagrams of complete data flow)
- **Section 11:** Integration Points for TorchRoyale DT (where model fits in)
- **Section 12:** Key Files Reference (quick lookup table)
- **Section 13:** Dependencies & Configuration
- **Section 14:** System Architecture Summary

**Best for:** Understanding complete system architecture, detailed component behavior, data transformations

---

### 2. INTEGRATION_POINTS.md (14 KB)
**Practical integration guide**

Focused on how the Decision Transformer model is integrated and how to use it:

- **Section 1:** Entry Points for DT Inference (video vs real-time)
- **Section 2:** Data Flow to DT Model (FrameState → Feature Vector → DT)
- **Section 3:** Model Configuration (YAML config, runtime overrides)
- **Section 4:** Context Window Management (rolling history, reset intervals)
- **Section 5:** Feature Encoding Details (97-dim breakdown)
- **Section 6:** Game State Data Sources (component outputs table)
- **Section 7:** Output Formats (recommendation objects, JSONL)
- **Section 8:** Fallback Behavior (when model unavailable)
- **Section 9:** Real-Time vs Batch Processing (performance characteristics)
- **Section 10:** Coordinate System for DT Output (tile grid layout)
- **Section 11:** Dependencies for DT Inference
- **Section 12:** Testing DT Integration (minimal test example)
- **Section 13:** Advanced: Custom DTInference Usage
- **Section 14:** Integration Checklist

**Best for:** Integrating DT into other systems, understanding data flow, writing code that uses DTStrategy

---

### 3. ARCHITECTURE_SUMMARY.txt (11 KB)
**Executive summary and quick reference**

High-level overview suitable for quick lookup:

- **Key Findings** (what the project does, architecture pattern, tech stack)
- **Directory Structure** (organized list of major components)
- **Critical Components** (6 key modules with 1-2 sentence descriptions)
- **Data Flow** (simplified flow diagrams for video and real-time)
- **Integration Points** (what's done vs what's missing)
- **Key File Reference** (quick lookup tables by category)
- **Performance Characteristics** (FPS, caching strategies, timing)
- **Dependencies** (organized by category)
- **Current Limitations & Gaps** (what doesn't exist yet)
- **What Works Today** (capabilities checklist)
- **Deployment Paths** (3 different usage modes)

**Best for:** Getting oriented quickly, presenting to others, finding specific components

---

## Quick Navigation

### I Want To...

**Understand the complete system**
→ Read: `ARCHITECTURE.md` (Sections 1-10)

**Integrate DT into my code**
→ Read: `INTEGRATION_POINTS.md` (Sections 1-3, 8-9)

**Find a specific component**
→ Read: `ARCHITECTURE_SUMMARY.txt` (Key File Reference)

**Set up model parameters**
→ Read: `INTEGRATION_POINTS.md` (Section 3)

**Understand data flow**
→ Read: `ARCHITECTURE.md` (Sections 4-6, 10) or `INTEGRATION_POINTS.md` (Section 2)

**Deploy live gameplay bot**
→ Read: `ARCHITECTURE.md` (Sections 2-3, 11) and `INTEGRATION_POINTS.md` (Section 9)

**Write a test**
→ Read: `INTEGRATION_POINTS.md` (Section 12)

**Understand coordinate system**
→ Read: `ARCHITECTURE.md` (Section 5) or `INTEGRATION_POINTS.md` (Section 10)

---

## System Architecture at a Glance

```
VIDEO INPUT
    ↓
FRAME PROCESSING (pipeline_live.py, video_analyzer.py)
    ├─ Detection Pipeline (Roboflow)
    ├─ OCR Pipeline (EasyOCR)
    └─ State Tracking (Game Phase, Elixir, Towers)
    ↓
GAME STATE (FrameState)
    ├─ player_elixir (0-10)
    ├─ opponent_elixir (calculated)
    ├─ game_phase (SINGLE/DOUBLE/TRIPLE/SUDDEN_DEATH)
    ├─ tower_health (OCR detected)
    ├─ detections (cards on field)
    └─ hand_cards (in your hand)
    ↓
FEATURE ENCODING (97-dim vector)
    ├─ Elixir (2)
    ├─ Phase (4)
    ├─ Time (1)
    ├─ Towers (6)
    ├─ Detections (80)
    └─ Hand (4)
    ↓
DECISION TRANSFORMER
    ├─ Input: Feature vector + target return
    ├─ Context: Rolling window of past states/actions
    └─ Output: (card_index, tile_position)
    ↓
RECOMMENDATION
    ├─ Card name (from hand)
    ├─ Tile X (0-17)
    └─ Tile Y (0-31, 17-31 is friendly side)
```

---

## Key Components Reference

| Component | File | Purpose |
|-----------|------|---------|
| **LivePipeline** | pipeline_live.py | Real-time (30 FPS) frame processing |
| **VideoAnalyzer** | src/video/video_analyzer.py | Batch video analysis → JSON |
| **DetectionPipeline** | detection_test.py | Roboflow YOLO inference |
| **DigitDetector** | src/ocr/digit_detector.py | EasyOCR digit reading |
| **GamePhaseTracker** | src/game_state/game_phase.py | Single/Double/Triple phase tracking |
| **OpponentElixirTracker** | src/recommendation/elixir_manager.py | Opponent elixir calculation |
| **CoordinateMapper** | src/grid/coordinate_mapper.py | Pixel ↔ tile grid conversion |
| **FeatureEncoder** | src/data/feature_encoder.py | FrameState → 97-dim vector |
| **DTStrategy** | src/recommendation/strategy.py | Decision Transformer wrapper |
| **DTInference** | src/transformer/inference.py | DT inference with context window |

---

## Data Types Reference

### FrameState (Complete Game State)
```python
{
    "timestamp_ms": int,
    "player_elixir": int (0-10),
    "opponent_elixir_estimated": float,
    "game_phase": str ("single"/"double"/"triple"/"sudden_death"),
    "game_time_remaining": Optional[int] (seconds),
    "detections": List[{
        "class_name": str,
        "tile_x": int (0-17),
        "tile_y": int (0-31),
        "is_opponent": bool,
        "is_on_field": bool
    }],
    "player_towers": Dict[str, {"hp_current": int, "health_percent": float}],
    "opponent_towers": Dict[str, {"hp_current": int, "health_percent": float}],
    "hand_cards": List[str]
}
```

### DT Recommendation
```python
(card_name: str, tile_x: int, tile_y: int)
# Example: ("hog-rider", 9, 24)
```

### Feature Vector
```python
np.ndarray(shape=(97,), dtype=np.float32)
# Normalized values in [0, 1] range
```

---

## Performance Summary

| Operation | Latency | Update Rate | Notes |
|-----------|---------|-------------|-------|
| OCR Detection | ~50ms | 10 Hz (every 3 frames) | Elixir, timer, multiplier |
| Roboflow Detection | ~100ms | 15 Hz (every 2 frames) | Cards and towers |
| Feature Encoding | ~5ms | 30 Hz (every frame) | 97-dim vector |
| DT Inference | ~10ms | 30 Hz (every frame) | With context window |
| Tower Health | ~50ms | 2 Hz (every 15 frames) | Full tower detection |
| Game Phase Update | <1ms | 30 Hz (every frame) | State machine |

**Overall:** 30 FPS target, 15-30 FPS typical with full pipeline

---

## Current Capabilities

### What Works
- Video analysis (MP4 → JSON game state)
- Real-time frame processing
- Card/tower detection (Roboflow torchroyale/4 model)
- Game state tracking (comprehensive)
- Decision Transformer inference
- Recommendation generation (JSONL output)
- Coordinate mapping (pixel ↔ tile grid)

### What's Missing
- ADB device integration (no screen capture, no input injection)
- GUI components (UI files are empty stubs)
- Real-time gameplay automation (needs ADB + event loop)

---

## Getting Started

### For Analysis (Works Today)
```python
from src.video.video_analyzer import VideoAnalyzer
from src.overlay.inference_runner import InferenceRunner

# Analyze video
analyzer = VideoAnalyzer()
result = analyzer.analyze_video("game.mp4")

# Generate recommendations with DT
runner = InferenceRunner(
    video_path="game.mp4",
    checkpoint_path="data/models/dt_checkpoint.pt",
    output_jsonl="recommendations.jsonl"
)
runner.run()
```

### For Real-Time (Mostly Ready)
```python
from pipeline_live import LivePipeline
from src.recommendation.strategy import DTStrategy

pipeline = LivePipeline(target_fps=30.0)
strategy = DTStrategy(checkpoint_path="data/models/dt_checkpoint.pt")
strategy.reset_game()

for frame in video_stream:
    state = pipeline.process_frame(frame)
    rec = strategy.recommend(state)
    if rec:
        card, tile_x, tile_y = rec
        # Handle recommendation
```

### For Live Gameplay (Needs ADB)
See ARCHITECTURE.md Section 3 for required additions

---

## File Locations

### Documentation
- `ARCHITECTURE.md` - Detailed technical documentation
- `INTEGRATION_POINTS.md` - DT integration guide
- `ARCHITECTURE_SUMMARY.txt` - Executive summary
- `DOCUMENTATION_INDEX.md` - This file

### Core Code
- `pipeline_live.py` - Real-time pipeline
- `src/video/video_analyzer.py` - Batch analysis
- `src/recommendation/strategy.py` - DT strategy wrapper
- `src/transformer/inference.py` - DT inference engine
- `src/data/feature_encoder.py` - Feature encoding

### Data & Configuration
- `data/models/dt_checkpoint.pt` - DT model weights
- `configs/inference.yaml` - Model configuration
- `src/constants/game_constants.py` - Game mechanics data

---

## Document Generation Date

Generated: April 21, 2025

For the latest information, see the source code in `/src/` directory.


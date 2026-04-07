# TorchRoyale - Comprehensive Project Implementation Plan

**Project:** Clash Royale Card Recommendation Assistant  
**Team:** Kashyap Sukshavasi, Johnathan Han, Albert Lungu  
**Client:** Steven Quast  
**Date Created:** January 13, 2026

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Phase 0: Environment Setup](#phase-0-environment-setup)
3. [Phase 1: Model Research & Integration](#phase-1-model-research--integration)
4. [Phase 2: Desktop Application Foundation](#phase-2-desktop-application-foundation)
5. [Phase 3: Game State Detection](#phase-3-game-state-detection)
6. [Phase 4: Recommendation Algorithm](#phase-4-recommendation-algorithm)
7. [Phase 5: Integration & Real-time Processing](#phase-5-integration--real-time-processing)
8. [Phase 6: Testing & Optimization](#phase-6-testing--optimization)
9. [Phase 7: Documentation & Presentation](#phase-7-documentation--presentation)
10. [Technical Architecture](#technical-architecture)
11. [Risk Mitigation Strategies](#risk-mitigation-strategies)

---

## Project Overview

### Goal
Create a desktop application that provides real-time card placement recommendations for Clash Royale by analyzing game screenshots and displaying optimal card choices and placement positions on an overlay grid.

### Core Constraints
- No direct game client interaction or automation
- Manual card placement by user
- Complies with Clash Royale fair play policies
- For research/practice use only, not live competitive matches

### Success Criteria
- Accurately detect cards in hand (>90% accuracy)
- Identify board state including unit positions
- Generate valid card placement recommendations
- Display recommendations in under 2 seconds
- Intuitive UI that doesn't obscure game view

---

## Phase 0: Environment Setup

**Duration:** 1-2 days  
**Priority:** Critical  
**Assigned To:** All team members

### Objectives
- Set up development environment for all team members
- Establish project structure
- Configure version control and collaboration tools
- Verify hardware capabilities

### Tasks

### Deliverables
- [x] Python 3.12 environment on all machines
- [x] Complete project structure created
- [x] All dependencies installed
- [x] Git repository configured with submodules
- [x] Roboflow API credentials secured
- [x] Hardware capabilities documented

### Testing Criteria
- Virtual environment activates without errors
- `python --version` returns 3.12.x
- `pip list` shows all required packages
- Can import torch, roboflow, tkinter without errors

---

## Phase 1: Model Research & Integration

**Duration:** 3-5 days  
**Priority:** Critical  
**Assigned To:** Kashyap + Albert

### Objectives
- Understand Roboflow model capabilities
- Test pre-trained model accuracy
- Set up model inference pipeline
- Determine if fine-tuning is necessary

### Tasks

#### 1.1 Roboflow Model Exploration
**What to do:**
Test the pre-trained Clash Royale card detection model from Roboflow

**How to do it:**
```python
# scripts/test_roboflow.py
from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv('private/.env')

# Initialize Roboflow
rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
project = rf.workspace().project("clash-royale-card-detection")
model = project.version(2).model

# Test on sample image
result = model.predict("path/to/test_image.jpg", confidence=40, overlap=30)

# Print results
print(result.json())

# Visualize
result.save("output.jpg")
```

**Expected Output:** 
- JSON response with detected cards
- Annotated image showing bounding boxes
- Confidence scores for each detection

#### 1.2 Model Accuracy Benchmarking
**What to do:**
Create test dataset and measure detection accuracy

**Test Categories:**
1. Card detection in hand (4 cards visible)
2. Card detection at different elixir levels
3. Card detection with visual effects (battles, explosions)
4. Board state detection (units, towers, king tower)
5. Arena detection (different arena backgrounds)

**How to do it:**
```python
# tests/test_detection.py
import pytest
from src.detection.card_detector import CardDetector

class TestCardDetection:
    def setup_method(self):
        self.detector = CardDetector()
        
    def test_hand_detection_accuracy(self):
        """Test detection of 4 cards in hand"""
        test_images = load_test_images('data/screenshots/hand_cards/')
        
        correct = 0
        total = 0
        
        for img_path, ground_truth in test_images:
            detected_cards = self.detector.detect_hand_cards(img_path)
            if detected_cards == ground_truth:
                correct += 1
            total += 1
            
        accuracy = correct / total
        assert accuracy >= 0.90, f"Accuracy {accuracy} below threshold"
        
    def test_board_unit_detection(self):
        """Test detection of units on board"""
        # Similar testing structure
        pass
```

**Success Criteria:**
- Hand card detection: >90% accuracy
- Board unit detection: >85% accuracy
- Tower detection: >95% accuracy
- Elixir count detection: >90% accuracy

**Expected Output:** Test report with accuracy metrics per category

#### 1.3 Model Inference Optimization
**What to do:**
Optimize model loading and inference speed for real-time performance

**How to do it:**
```python
# src/detection/model_loader.py
import torch
from roboflow import Roboflow
import os
from dotenv import load_dotenv

class ModelLoader:
    def __init__(self):
        load_dotenv('private/.env')
        self.api_key = os.getenv('ROBOFLOW_API_KEY')
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, cache=True):
        """Load Roboflow model with caching"""
        if cache and self.model is not None:
            return self.model
            
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace().project("clash-royale-card-detection")
        self.model = project.version(2).model
        
        return self.model
        
    def warm_up(self):
        """Perform warm-up inference to optimize first prediction"""
        dummy_image = self._create_dummy_image()
        self.model.predict(dummy_image)
```

**Performance Targets:**
- Model load time: <5 seconds
- First inference: <1 second
- Subsequent inferences: <500ms
- Memory usage: <2GB

**Expected Output:** Optimized model loader with performance metrics

#### 1.4 Card Detection Pipeline
**What to do:**
Create unified pipeline for detecting all game elements

**How to do it:**
```python
# src/detection/card_detector.py
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image

class CardDetector:
    def __init__(self, model_loader):
        self.model = model_loader.load_model()
        self.card_regions = {
            'hand': (0, 800, 1920, 1080),  # Bottom of screen
            'board': (0, 200, 1920, 800),   # Middle section
            'elixir': (1600, 900, 1800, 1000)  # Bottom right
        }
        
    def detect_hand_cards(self, image: np.ndarray) -> List[Dict]:
        """
        Detect cards in player's hand
        
        Returns:
            List of dicts with 'card_name', 'confidence', 'bbox'
        """
        hand_region = self._crop_region(image, self.card_regions['hand'])
        predictions = self.model.predict(hand_region, confidence=60)
        
        cards = []
        for pred in predictions:
            cards.append({
                'card_name': pred['class'],
                'confidence': pred['confidence'],
                'bbox': pred['bbox'],
                'position_index': self._determine_hand_position(pred['bbox'])
            })
        
        return sorted(cards, key=lambda x: x['position_index'])
    
    def detect_board_units(self, image: np.ndarray) -> List[Dict]:
        """Detect units currently on the board"""
        board_region = self._crop_region(image, self.card_regions['board'])
        predictions = self.model.predict(board_region, confidence=50)
        
        units = []
        for pred in predictions:
            units.append({
                'unit_type': pred['class'],
                'position': self._bbox_to_grid_position(pred['bbox']),
                'side': self._determine_side(pred['bbox']),  # 'friendly' or 'enemy'
                'confidence': pred['confidence']
            })
        
        return units
    
    def detect_elixir(self, image: np.ndarray) -> int:
        """Detect current elixir count"""
        # OCR or template matching for elixir counter
        elixir_region = self._crop_region(image, self.card_regions['elixir'])
        # Implementation using pytesseract or template matching
        pass
    
    def _crop_region(self, image: np.ndarray, region: Tuple[int, int, int, int]):
        """Crop image to specific region"""
        x1, y1, x2, y2 = region
        return image[y1:y2, x1:x2]
    
    def _bbox_to_grid_position(self, bbox: Dict) -> Tuple[int, int]:
        """Convert bounding box to grid coordinates"""
        # Arena is 18x32 tiles
        # Convert pixel coordinates to tile coordinates
        pass
```

**Expected Output:** Robust detection pipeline for all game elements

#### 1.5 Fine-tuning Decision
**What to do:**
Determine if model fine-tuning is necessary based on accuracy results

**Decision Tree:**
- If accuracy >90% on all categories: Use pre-trained model as-is
- If accuracy 80-90%: Fine-tune on edge cases
- If accuracy <80%: Full fine-tuning required

**If Fine-tuning Required:**
```python
# scripts/fine_tune_model.py
# 1. Prepare training dataset
# 2. Set up PyTorch training loop
# 3. Use transfer learning from Roboflow base model
# 4. Train for 10-20 epochs
# 5. Validate on hold-out test set
# 6. Export fine-tuned weights
```

**Expected Output:** Decision document with accuracy justification

#### 1.6 Card Metadata Database
**What to do:**
Create database of all Clash Royale cards with stats

**How to do it:**
```python
# data/card_data/cards.json
{
  "knight": {
    "name": "Knight",
    "elixir_cost": 3,
    "type": "troop",
    "rarity": "common",
    "target": "ground",
    "speed": "medium",
    "range": "melee",
    "deploy_time": 1,
    "counters": ["skeleton_army", "guards"],
    "countered_by": ["mini_pekka", "valkyrie"]
  },
  "fireball": {
    "name": "Fireball",
    "elixir_cost": 4,
    "type": "spell",
    "rarity": "rare",
    "radius": 2.5,
    "damage": 572,
    "counters": ["witch", "wizard", "musketeer"],
    "usage": "offensive_defensive"
  }
  // ... all cards
}
```

**Data Sources:**
- Official Clash Royale API
- RoyaleAPI (https://royaleapi.com/)
- Manual compilation from game

**Expected Output:** Complete card database in JSON format

### Deliverables
- [ ] Roboflow model integrated and tested
- [ ] Accuracy benchmarks documented
- [ ] Optimized inference pipeline
- [ ] Card detection module complete
- [ ] Fine-tuning decision made and documented
- [ ] Card metadata database created

### Testing Criteria
- All detection tests pass with >90% accuracy
- Inference time meets <500ms target
- Can detect cards in various game scenarios
- Card database has all current cards (107+ cards)

---

## Phase 2: Desktop Application Foundation

**Duration:** 3-4 days  
**Priority:** High  
**Assigned To:** Johnathan + Albert

### Objectives
- Create basic Tkinter application window
- Implement grid overlay system
- Set up card display mechanism
- Design user-friendly interface

### Tasks

#### 2.1 Main Application Window
**What to do:**
Create the main Tkinter window with basic layout

**How to do it:**
```python
# src/ui/main_window.py
import tkinter as tk
from tkinter import ttk
from typing import Optional

class TorchRoyaleApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TorchRoyale - Card Recommendation Assistant")
        self.root.geometry("800x600")
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self._setup_ui()
        self._setup_menu()
        
    def _setup_ui(self):
        """Setup main UI components"""
        # Control panel (top)
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N))
        
        # Game board display (center)
        self.board_frame = ttk.Frame(self.root, padding="10")
        self.board_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar (bottom)
        self.status_frame = ttk.Frame(self.root, padding="5")
        self.status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self._create_controls()
        self._create_board()
        self._create_status_bar()
        
    def _setup_menu(self):
        """Setup application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings", command=self._open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Grid", command=self._toggle_grid)
        view_menu.add_checkbutton(label="Show Confidence", command=self._toggle_confidence)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._open_docs)
        help_menu.add_command(label="About", command=self._show_about)
        
    def _create_controls(self):
        """Create control panel with buttons"""
        # Start/Stop button
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Detection",
            command=self._toggle_detection
        )
        self.start_button.grid(row=0, column=0, padx=5)
        
        # Screenshot button
        self.screenshot_button = ttk.Button(
            self.control_frame,
            text="Take Screenshot",
            command=self._take_screenshot
        )
        self.screenshot_button.grid(row=0, column=1, padx=5)
        
        # Status indicator
        self.status_label = ttk.Label(
            self.control_frame,
            text="Status: Idle",
            foreground="blue"
        )
        self.status_label.grid(row=0, column=2, padx=20)
        
    def _create_board(self):
        """Create game board canvas"""
        self.canvas = tk.Canvas(
            self.board_frame,
            width=600,
            height=400,
            bg='#2e7d32'  # Green arena color
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def _create_status_bar(self):
        """Create status bar at bottom"""
        self.status_text = tk.StringVar()
        self.status_text.set("Ready")
        
        status_label = ttk.Label(
            self.status_frame,
            textvariable=self.status_text,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_label.pack(fill=tk.X)
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = TorchRoyaleApp()
    app.run()
```

**Expected Output:** Functional Tkinter window with basic controls

#### 2.2 Grid Overlay System
**What to do:**
Implement the 18x32 tile grid overlay matching Clash Royale arena

**How to do it:**
```python
# src/ui/grid_overlay.py
import tkinter as tk
from typing import Tuple

class GridOverlay:
    def __init__(self, canvas: tk.Canvas, width: int, height: int):
        self.canvas = canvas
        self.width = width
        self.height = height
        
        # Clash Royale arena is 18 tiles wide, 32 tiles tall
        self.cols = 18
        self.rows = 32
        
        # Calculate tile dimensions
        self.tile_width = width / self.cols
        self.tile_height = height / self.rows
        
        self.grid_lines = []
        self.grid_visible = True
        
        self._draw_grid()
        
    def _draw_grid(self):
        """Draw grid lines on canvas"""
        # Vertical lines
        for i in range(self.cols + 1):
            x = i * self.tile_width
            line = self.canvas.create_line(
                x, 0, x, self.height,
                fill='white',
                width=1,
                dash=(2, 4)
            )
            self.grid_lines.append(line)
        
        # Horizontal lines
        for i in range(self.rows + 1):
            y = i * self.tile_height
            line = self.canvas.create_line(
                0, y, self.width, y,
                fill='white',
                width=1,
                dash=(2, 4)
            )
            self.grid_lines.append(line)
        
        # Draw river dividing line (middle)
        river_y = (self.rows // 2) * self.tile_height
        river_line = self.canvas.create_line(
            0, river_y, self.width, river_y,
            fill='cyan',
            width=3
        )
        self.grid_lines.append(river_line)
        
        # Draw tower positions
        self._draw_towers()
        
    def _draw_towers(self):
        """Draw tower position indicators"""
        # King towers (top and bottom center)
        self._draw_tower_marker(self.cols // 2, 2, 'red')  # Enemy
        self._draw_tower_marker(self.cols // 2, self.rows - 3, 'blue')  # Friendly
        
        # Princess towers
        self._draw_tower_marker(4, 7, 'red')  # Enemy left
        self._draw_tower_marker(self.cols - 5, 7, 'red')  # Enemy right
        self._draw_tower_marker(4, self.rows - 8, 'blue')  # Friendly left
        self._draw_tower_marker(self.cols - 5, self.rows - 8, 'blue')  # Friendly right
        
    def _draw_tower_marker(self, col: int, row: int, color: str):
        """Draw a tower marker at grid position"""
        x = col * self.tile_width
        y = row * self.tile_height
        
        marker = self.canvas.create_oval(
            x - 10, y - 10, x + 10, y + 10,
            fill=color,
            outline='white',
            width=2
        )
        self.grid_lines.append(marker)
        
    def pixel_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        """Convert pixel coordinates to grid coordinates"""
        col = int(x / self.tile_width)
        row = int(y / self.tile_height)
        return (col, row)
        
    def grid_to_pixel(self, col: int, row: int) -> Tuple[int, int]:
        """Convert grid coordinates to pixel coordinates (center of tile)"""
        x = (col + 0.5) * self.tile_width
        y = (row + 0.5) * self.tile_height
        return (int(x), int(y))
        
    def highlight_tile(self, col: int, row: int, color: str = 'yellow'):
        """Highlight a specific tile"""
        x, y = self.grid_to_pixel(col, row)
        
        rect = self.canvas.create_rectangle(
            x - self.tile_width/2, y - self.tile_height/2,
            x + self.tile_width/2, y + self.tile_height/2,
            fill=color,
            stipple='gray50',
            outline=color,
            width=3
        )
        
        return rect
        
    def toggle_visibility(self):
        """Toggle grid line visibility"""
        self.grid_visible = not self.grid_visible
        state = tk.NORMAL if self.grid_visible else tk.HIDDEN
        
        for line in self.grid_lines:
            self.canvas.itemconfig(line, state=state)
```

**Expected Output:** Interactive grid overlay with tile highlighting

#### 2.3 Card Display System
**What to do:**
Create system to display recommended cards on the overlay

**How to do it:**
```python
# src/ui/card_display.py
import tkinter as tk
from PIL import Image, ImageTk
from typing import Optional, Tuple

class CardDisplay:
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.card_images = {}
        self.active_displays = []
        
        self._load_card_images()
        
    def _load_card_images(self):
        """Load card images from assets"""
        # Load card images from data/card_images/
        # Resize to appropriate size for display
        pass
        
    def show_recommendation(
        self,
        card_name: str,
        grid_pos: Tuple[int, int],
        confidence: float
    ):
        """
        Display recommended card at grid position
        
        Args:
            card_name: Name of the card to display
            grid_pos: (col, row) grid coordinates
            confidence: Confidence score 0-1
        """
        # Clear previous recommendations
        self.clear_recommendations()
        
        # Get pixel position from grid
        x, y = self._grid_to_canvas_coords(grid_pos)
        
        # Create pulsing highlight circle
        highlight = self.canvas.create_oval(
            x - 30, y - 30, x + 30, y + 30,
            fill='yellow',
            outline='orange',
            width=3
        )
        self.active_displays.append(highlight)
        
        # Display card image
        if card_name in self.card_images:
            card_img = self.canvas.create_image(
                x, y,
                image=self.card_images[card_name],
                anchor=tk.CENTER
            )
            self.active_displays.append(card_img)
        
        # Display card name and confidence
        text = f"{card_name}\n{confidence:.0%}"
        label = self.canvas.create_text(
            x, y + 40,
            text=text,
            fill='white',
            font=('Arial', 10, 'bold'),
            justify=tk.CENTER
        )
        self.active_displays.append(label)
        
        # Animate highlight (optional)
        self._animate_highlight(highlight)
        
    def clear_recommendations(self):
        """Clear all displayed recommendations"""
        for item in self.active_displays:
            self.canvas.delete(item)
        self.active_displays = []
        
    def _animate_highlight(self, item_id: int):
        """Create pulsing animation for highlight"""
        # Tkinter animation using after() method
        pass
```

**Expected Output:** Card recommendation display system

#### 2.4 Window Management & Positioning
**What to do:**
Add functionality to position window alongside game client

**How to do it:**
```python
# Extension to main_window.py
def _open_settings(self):
    """Open settings dialog"""
    settings_window = tk.Toplevel(self.root)
    settings_window.title("Settings")
    settings_window.geometry("400x300")
    
    # Window positioning
    position_frame = ttk.LabelFrame(settings_window, text="Window Position", padding=10)
    position_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Button(
        position_frame,
        text="Left of Game",
        command=lambda: self._position_window('left')
    ).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(
        position_frame,
        text="Right of Game",
        command=lambda: self._position_window('right')
    ).pack(side=tk.LEFT, padx=5)
    
    # Transparency slider
    transparency_frame = ttk.LabelFrame(settings_window, text="Transparency", padding=10)
    transparency_frame.pack(fill=tk.X, padx=10, pady=5)
    
    self.transparency_var = tk.DoubleVar(value=1.0)
    transparency_slider = ttk.Scale(
        transparency_frame,
        from_=0.3,
        to=1.0,
        variable=self.transparency_var,
        command=self._update_transparency
    )
    transparency_slider.pack(fill=tk.X)

def _position_window(self, position: str):
    """Position window relative to game"""
    screen_width = self.root.winfo_screenwidth()
    window_width = 800
    
    if position == 'left':
        x = 0
    elif position == 'right':
        x = screen_width - window_width
    else:
        x = (screen_width - window_width) // 2
    
    self.root.geometry(f"{window_width}x600+{x}+0")

def _update_transparency(self, value):
    """Update window transparency"""
    self.root.attributes('-alpha', float(value))
```

**Expected Output:** Configurable window positioning and transparency

### Deliverables
- [ ] Main Tkinter application window
- [ ] 18x32 grid overlay system
- [ ] Card display mechanism
- [ ] Window positioning controls
- [ ] Settings dialog

### Testing Criteria
- Application launches without errors
- Grid displays correctly with proper proportions
- Can highlight tiles and display cards
- Window positioning works on test machine
- UI is responsive and intuitive

---

## Phase 3: Game State Detection

**Duration:** 4-5 days  
**Priority:** Critical  
**Assigned To:** Kashyap + Johnathan

### Objectives
- Implement screen capture system
- Process screenshots in real-time
- Detect complete game state
- Handle edge cases and errors

### Tasks

#### 3.1 Screen Capture System
**What to do:**
Implement fast, reliable screenshot capture of game window

**How to do it:**
```python
# src/capture/screen_capture.py
import mss
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import time

class ScreenCapture:
    def __init__(self, target_fps: int = 2):
        self.sct = mss.mss()
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_capture_time = 0
        
        # Define capture region (full screen or specific window)
        self.monitor = None
        self._detect_game_window()
        
    def _detect_game_window(self):
        """Detect Clash Royale window position and size"""
        # Try to find window by title
        # For now, use primary monitor
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
    def set_capture_region(self, x: int, y: int, width: int, height: int):
        """Manually set capture region"""
        self.monitor = {
            'left': x,
            'top': y,
            'width': width,
            'height': height
        }
        
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture screenshot with frame rate limiting
        
        Returns:
            numpy array of captured image, or None if rate limited
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_capture_time < self.frame_interval:
            return None
            
        # Capture screenshot
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to RGB
        img = img[:, :, :3]
        
        self.last_capture_time = current_time
        
        return img
        
    def capture_continuous(self, callback):
        """
        Continuously capture screenshots and call callback
        
        Args:
            callback: Function to call with each captured image
        """
        try:
            while True:
                img = self.capture()
                if img is not None:
                    callback(img)
                    
                time.sleep(0.01)  # Small sleep to prevent CPU overload
                
        except KeyboardInterrupt:
            print("Capture stopped")
            
    def save_screenshot(self, filepath: str):
        """Save a single screenshot to file"""
        img = self.capture()
        if img is not None:
            Image.fromarray(img).save(filepath)
            return True
        return False
```

**Expected Output:** Fast screen capture at 2 FPS

#### 3.2 Game State Manager
**What to do:**
Create centralized system to track complete game state

**How to do it:**
```python
# src/detection/game_state.py
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Side(Enum):
    FRIENDLY = "friendly"
    ENEMY = "enemy"

@dataclass
class Card:
    name: str
    elixir_cost: int
    position_index: int  # 0-3 for hand position
    confidence: float

@dataclass
class Unit:
    unit_type: str
    grid_position: tuple[int, int]
    side: Side
    health_percent: float
    confidence: float

@dataclass
class Tower:
    tower_type: str  # 'king', 'princess_left', 'princess_right'
    side: Side
    health_percent: float
    is_destroyed: bool

@dataclass
class GameState:
    # Player state
    hand_cards: List[Card]
    elixir_current: int
    elixir_max: int
    
    # Board state
    friendly_units: List[Unit]
    enemy_units: List[Unit]
    
    # Tower state
    towers: List[Tower]
    
    # Game info
    game_time: Optional[int]  # Seconds remaining
    is_overtime: bool
    
    # Metadata
    timestamp: float
    confidence_score: float

class GameStateManager:
    def __init__(self, card_detector, board_analyzer):
        self.card_detector = card_detector
        self.board_analyzer = board_analyzer
        self.current_state = None
        self.state_history = []
        
    def update_state(self, screenshot: np.ndarray) -> GameState:
        """
        Update game state from new screenshot
        
        Args:
            screenshot: numpy array of game screenshot
            
        Returns:
            Updated GameState object
        """
        # Detect hand cards
        hand_cards = self.card_detector.detect_hand_cards(screenshot)
        
        # Detect board units
        units = self.board_analyzer.detect_units(screenshot)
        
        # Separate friendly and enemy units
        friendly_units = [u for u in units if u.side == Side.FRIENDLY]
        enemy_units = [u for u in units if u.side == Side.ENEMY]
        
        # Detect towers
        towers = self.board_analyzer.detect_towers(screenshot)
        
        # Detect elixir
        elixir_current, elixir_max = self.card_detector.detect_elixir(screenshot)
        
        # Create game state
        state = GameState(
            hand_cards=hand_cards,
            elixir_current=elixir_current,
            elixir_max=elixir_max,
            friendly_units=friendly_units,
            enemy_units=enemy_units,
            towers=towers,
            game_time=None,  # TODO: Implement timer detection
            is_overtime=False,
            timestamp=time.time(),
            confidence_score=self._calculate_confidence(hand_cards, units, towers)
        )
        
        # Store state
        self.current_state = state
        self.state_history.append(state)
        
        # Keep only recent history (last 30 seconds)
        self._trim_history()
        
        return state
        
    def _calculate_confidence(self, cards, units, towers):
        """Calculate overall confidence score for detection"""
        if not cards:
            return 0.0
            
        scores = [c.confidence for c in cards]
        scores += [u.confidence for u in units]
        
        return sum(scores) / len(scores) if scores else 0.0
        
    def _trim_history(self, max_states: int = 60):
        """Keep only recent history"""
        if len(self.state_history) > max_states:
            self.state_history = self.state_history[-max_states:]
            
    def get_state_changes(self) -> dict:
        """Detect changes between current and previous state"""
        if len(self.state_history) < 2:
            return {}
            
        prev_state = self.state_history[-2]
        curr_state = self.current_state
        
        changes = {
            'new_enemy_units': [],
            'destroyed_towers': [],
            'elixir_change': curr_state.elixir_current - prev_state.elixir_current
        }
        
        # Detect new enemy units
        prev_enemy_types = {u.unit_type for u in prev_state.enemy_units}
        for unit in curr_state.enemy_units:
            if unit.unit_type not in prev_enemy_types:
                changes['new_enemy_units'].append(unit)
        
        return changes
```

**Expected Output:** Complete game state tracking system

#### 3.3 Board Analysis Module
**What to do:**
Implement advanced board analysis for unit positioning and threats

**How to do it:**
```python
# src/detection/board_analyzer.py
import numpy as np
from typing import List, Tuple, Dict
from .game_state import Unit, Tower, Side

class BoardAnalyzer:
    def __init__(self, model):
        self.model = model
        self.grid_size = (18, 32)  # cols, rows
        
    def detect_units(self, screenshot: np.ndarray) -> List[Unit]:
        """Detect all units on the board"""
        predictions = self.model.predict(screenshot, confidence=50)
        
        units = []
        for pred in predictions:
            # Filter out cards in hand
            if self._is_in_hand_region(pred['bbox']):
                continue
                
            unit = Unit(
                unit_type=pred['class'],
                grid_position=self._bbox_to_grid(pred['bbox']),
                side=self._determine_side(pred['bbox']),
                health_percent=1.0,  # TODO: Implement health detection
                confidence=pred['confidence']
            )
            units.append(unit)
            
        return units
        
    def detect_towers(self, screenshot: np.ndarray) -> List[Tower]:
        """Detect all towers and their health"""
        # Known tower positions
        tower_positions = {
            'king_enemy': (9, 2),
            'princess_left_enemy': (4, 7),
            'princess_right_enemy': (14, 7),
            'king_friendly': (9, 30),
            'princess_left_friendly': (4, 25),
            'princess_right_friendly': (14, 25)
        }
        
        towers = []
        for tower_name, grid_pos in tower_positions.items():
            # Check if tower exists in screenshot
            tower_region = self._get_tower_region(screenshot, grid_pos)
            
            # Detect tower presence and health
            is_destroyed = self._is_tower_destroyed(tower_region)
            health = self._estimate_tower_health(tower_region) if not is_destroyed else 0.0
            
            side = Side.ENEMY if 'enemy' in tower_name else Side.FRIENDLY
            tower_type = tower_name.split('_')[0]
            
            tower = Tower(
                tower_type=tower_type,
                side=side,
                health_percent=health,
                is_destroyed=is_destroyed
            )
            towers.append(tower)
            
        return towers
        
    def analyze_threat_level(self, state: 'GameState') -> Dict[str, float]:
        """
        Analyze current threat level from enemy units
        
        Returns:
            Dict with threat levels for different lanes
        """
        threats = {
            'left_lane': 0.0,
            'center_lane': 0.0,
            'right_lane': 0.0,
            'overall': 0.0
        }
        
        for unit in state.enemy_units:
            col, row = unit.grid_position
            
            # Determine lane (left: 0-5, center: 6-11, right: 12-17)
            if col < 6:
                lane = 'left_lane'
            elif col < 12:
                lane = 'center_lane'
            else:
                lane = 'right_lane'
            
            # Threat increases as unit gets closer to friendly side
            distance_factor = (32 - row) / 32  # Higher value = closer to friendly
            unit_threat = self._get_unit_threat_value(unit.unit_type)
            
            threats[lane] += unit_threat * distance_factor
            
        threats['overall'] = sum([threats['left_lane'], threats['center_lane'], threats['right_lane']])
        
        return threats
        
    def _get_unit_threat_value(self, unit_type: str) -> float:
        """Get base threat value for unit type"""
        threat_values = {
            'giant': 0.8,
            'pekka': 0.9,
            'hog_rider': 0.7,
            'goblin_barrel': 0.6,
            'minion_horde': 0.5,
            # ... more units
        }
        return threat_values.get(unit_type, 0.5)
        
    def _bbox_to_grid(self, bbox: Dict) -> Tuple[int, int]:
        """Convert bounding box to grid coordinates"""
        center_x = (bbox['x1'] + bbox['x2']) / 2
        center_y = (bbox['y1'] + bbox['y2']) / 2
        
        # Assuming screenshot is 1920x1080, arena is centered
        arena_width = 1080  # Arena is square-ish
        arena_height = 1920
        
        col = int((center_x / arena_width) * self.grid_size[0])
        row = int((center_y / arena_height) * self.grid_size[1])
        
        return (col, row)
        
    def _determine_side(self, bbox: Dict) -> Side:
        """Determine if unit is friendly or enemy based on position"""
        center_y = (bbox['y1'] + bbox['y2']) / 2
        
        # Enemy units are in top half, friendly in bottom half
        if center_y < 960:  # Half of 1920
            return Side.ENEMY
        else:
            return Side.FRIENDLY
```

**Expected Output:** Comprehensive board analysis capabilities

#### 3.4 Error Handling & Edge Cases
**What to do:**
Handle detection failures, occlusions, and edge cases

**How to do it:**
```python
# src/detection/error_handler.py
from typing import Optional
from .game_state import GameState
import logging

logger = logging.getLogger(__name__)

class DetectionErrorHandler:
    def __init__(self):
        self.error_count = 0
        self.max_errors = 5
        self.last_valid_state = None
        
    def validate_state(self, state: GameState) -> bool:
        """
        Validate that detected game state is reasonable
        
        Returns:
            True if state is valid, False otherwise
        """
        # Check hand cards (should be 4)
        if len(state.hand_cards) not in [3, 4]:
            logger.warning(f"Invalid hand card count: {len(state.hand_cards)}")
            return False
            
        # Check elixir (should be 0-10)
        if not (0 <= state.elixir_current <= state.elixir_max):
            logger.warning(f"Invalid elixir: {state.elixir_current}/{state.elixir_max}")
            return False
            
        # Check confidence score
        if state.confidence_score < 0.5:
            logger.warning(f"Low confidence: {state.confidence_score}")
            return False
            
        # Check tower count (should be 6 initially)
        active_towers = [t for t in state.towers if not t.is_destroyed]
        if len(active_towers) < 2:  # At least king towers should exist
            logger.warning(f"Invalid tower count: {len(active_towers)}")
            return False
            
        return True
        
    def handle_detection_failure(self, error: Exception) -> Optional[GameState]:
        """
        Handle detection failure
        
        Returns:
            Last valid state or None
        """
        self.error_count += 1
        logger.error(f"Detection failed ({self.error_count}/{self.max_errors}): {error}")
        
        if self.error_count >= self.max_errors:
            logger.critical("Too many detection failures, stopping detection")
            return None
            
        # Return last valid state as fallback
        return self.last_valid_state
        
    def reset_errors(self):
        """Reset error count after successful detection"""
        self.error_count = 0
```

**Expected Output:** Robust error handling system

### Deliverables
- [ ] Screen capture system working
- [ ] Game state manager implemented
- [ ] Board analysis module complete
- [ ] Error handling in place
- [ ] Edge case testing completed

### Testing Criteria
- Can capture screenshots at 2 FPS consistently
- Detects all hand cards correctly (>90% accuracy)
- Identifies friendly vs enemy units correctly
- Handles occlusions and visual effects
- Error recovery works as expected

---

## Phase 4: Recommendation Algorithm

**Duration:** 5-7 days  
**Priority:** Critical  
**Assigned To:** All team members

### Objectives
- Design card recommendation strategy
- Implement counter card logic
- Create elixir management system
- Develop placement optimization

### Tasks

#### 4.1 Strategy Framework
**What to do:**
Create framework for different strategic approaches

**How to do it:**
```python
# src/recommendation/strategy.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from src.detection.game_state import GameState, Card, Unit

class Strategy(ABC):
    @abstractmethod
    def recommend(self, state: GameState) -> Optional[Tuple[Card, Tuple[int, int]]]:
        """
        Generate card recommendation
        
        Returns:
            Tuple of (card_to_play, grid_position) or None
        """
        pass

class DefensiveStrategy(Strategy):
    """Focus on defending against enemy pushes"""
    
    def recommend(self, state: GameState) -> Optional[Tuple[Card, Tuple[int, int]]]:
        # Priority: Counter enemy threats
        if state.enemy_units:
            return self._counter_threat(state)
        return None
        
    def _counter_threat(self, state: GameState):
        # Find biggest threat
        highest_threat_unit = max(
            state.enemy_units,
            key=lambda u: self._threat_score(u, state)
        )
        
        # Find counter card in hand
        counter_card = self._find_counter_card(
            highest_threat_unit.unit_type,
            state.hand_cards,
            state.elixir_current
        )
        
        if counter_card:
            # Determine placement position
            placement_pos = self._optimal_defense_position(
                highest_threat_unit,
                counter_card
            )
            return (counter_card, placement_pos)
            
        return None

class OffensiveStrategy(Strategy):
    """Focus on attacking enemy towers"""
    
    def recommend(self, state: GameState) -> Optional[Tuple[Card, Tuple[int, int]]]:
        # Priority: Build push when elixir is high
        if state.elixir_current >= 7:
            return self._build_push(state)
        return None
        
    def _build_push(self, state: GameState):
        # Find best offensive card
        offensive_card = self._find_best_offensive_card(
            state.hand_cards,
            state.elixir_current
        )
        
        if offensive_card:
            # Determine push lane
            push_lane = self._select_push_lane(state)
            placement_pos = self._offensive_placement(push_lane)
            return (offensive_card, placement_pos)
            
        return None

class BalancedStrategy(Strategy):
    """Balance between offense and defense"""
    
    def __init__(self):
        self.defensive = DefensiveStrategy()
        self.offensive = OffensiveStrategy()
        
    def recommend(self, state: GameState) -> Optional[Tuple[Card, Tuple[int, int]]]:
        # Calculate threat level
        threat_level = self._calculate_threat_level(state)
        
        # Defend if under threat
        if threat_level > 0.6:
            return self.defensive.recommend(state)
            
        # Attack if opportunity exists
        if state.elixir_current >= 6 and threat_level < 0.3:
            return self.offensive.recommend(state)
            
        # Default to defensive
        return self.defensive.recommend(state)
        
    def _calculate_threat_level(self, state: GameState) -> float:
        """Calculate current threat level 0-1"""
        if not state.enemy_units:
            return 0.0
            
        threat_sum = sum(
            self._unit_threat_value(u.unit_type) * (1 - u.grid_position[1] / 32)
            for u in state.enemy_units
        )
        
        return min(threat_sum / 5.0, 1.0)  # Normalize to 0-1

class StrategyManager:
    def __init__(self, strategy_type: str = 'balanced'):
        self.strategies = {
            'defensive': DefensiveStrategy(),
            'offensive': OffensiveStrategy(),
            'balanced': BalancedStrategy()
        }
        self.current_strategy = self.strategies[strategy_type]
        
    def set_strategy(self, strategy_type: str):
        """Change active strategy"""
        self.current_strategy = self.strategies[strategy_type]
        
    def get_recommendation(self, state: GameState) -> Optional[Tuple[Card, Tuple[int, int]]]:
        """Get recommendation from current strategy"""
        return self.current_strategy.recommend(state)
```

**Expected Output:** Flexible strategy system

#### 4.2 Counter Card Database
**What to do:**
Create comprehensive counter card relationships

**How to do it:**
```python
# src/recommendation/counter_cards.py
from typing import List, Dict, Set

class CounterDatabase:
    def __init__(self):
        self.counters = self._build_counter_database()
        
    def _build_counter_database(self) -> Dict[str, List[Dict]]:
        """
        Build database of counter relationships
        
        Structure:
        {
            'unit_type': [
                {'card': 'counter_card', 'effectiveness': 0.9, 'elixir_trade': 2},
                ...
            ]
        }
        """
        return {
            'giant': [
                {'card': 'mini_pekka', 'effectiveness': 0.9, 'elixir_trade': 1},
                {'card': 'inferno_tower', 'effectiveness': 0.95, 'elixir_trade': 0},
                {'card': 'skeleton_army', 'effectiveness': 0.85, 'elixir_trade': 4},
            ],
            'hog_rider': [
                {'card': 'cannon', 'effectiveness': 0.8, 'elixir_trade': 1},
                {'card': 'tornado', 'effectiveness': 0.7, 'elixir_trade': 1},
                {'card': 'mini_pekka', 'effectiveness': 0.9, 'elixir_trade': 0},
            ],
            'minion_horde': [
                {'card': 'arrows', 'effectiveness': 1.0, 'elixir_trade': 2},
                {'card': 'fireball', 'effectiveness': 0.95, 'elixir_trade': 1},
                {'card': 'wizard', 'effectiveness': 0.85, 'elixir_trade': 0},
            ],
            'balloon': [
                {'card': 'musketeer', 'effectiveness': 0.8, 'elixir_trade': 1},
                {'card': 'inferno_dragon', 'effectiveness': 0.9, 'elixir_trade': 1},
                {'card': 'tornado', 'effectiveness': 0.7, 'elixir_trade': 2},
            ],
            'elite_barbarians': [
                {'card': 'valkyrie', 'effectiveness': 0.8, 'elixir_trade': 2},
                {'card': 'skeleton_army', 'effectiveness': 0.85, 'elixir_trade': 3},
                {'card': 'bowler', 'effectiveness': 0.9, 'elixir_trade': 1},
            ],
            'goblin_barrel': [
                {'card': 'log', 'effectiveness': 1.0, 'elixir_trade': 1},
                {'card': 'arrows', 'effectiveness': 0.95, 'elixir_trade': 0},
                {'card': 'zap', 'effectiveness': 0.9, 'elixir_trade': 1},
            ],
            'witch': [
                {'card': 'fireball', 'effectiveness': 0.9, 'elixir_trade': 1},
                {'card': 'valkyrie', 'effectiveness': 0.85, 'elixir_trade': 1},
                {'card': 'knight', 'effectiveness': 0.75, 'elixir_trade': 2},
            ],
            # Add all card counters...
        }
        
    def get_counters(self, unit_type: str, available_cards: List['Card']) -> List[Dict]:
        """
        Get available counter cards for a unit
        
        Args:
            unit_type: Type of enemy unit to counter
            available_cards: Cards currently in hand
            
        Returns:
            List of counter options sorted by effectiveness
        """
        if unit_type not in self.counters:
            return []
            
        available_card_names = {card.name for card in available_cards}
        
        # Filter to available cards
        counters = [
            counter for counter in self.counters[unit_type]
            if counter['card'] in available_card_names
        ]
        
        # Sort by effectiveness
        counters.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        return counters
        
    def get_best_counter(
        self,
        unit_type: str,
        available_cards: List['Card'],
        available_elixir: int
    ) -> Optional['Card']:
        """
        Get single best counter card considering elixir
        
        Returns:
            Best counter card or None
        """
        counters = self.get_counters(unit_type, available_cards)
        
        # Filter by elixir cost
        affordable_counters = [
            c for c in counters
            if self._get_card_cost(c['card']) <= available_elixir
        ]
        
        if not affordable_counters:
            return None
            
        # Return best affordable counter
        best_counter_name = affordable_counters[0]['card']
        
        for card in available_cards:
            if card.name == best_counter_name:
                return card
                
        return None
        
    def _get_card_cost(self, card_name: str) -> int:
        """Get elixir cost for card"""
        # Load from card database
        from src.utils.config import load_card_data
        card_data = load_card_data()
        return card_data[card_name]['elixir_cost']
```

**Expected Output:** Complete counter card database

#### 4.3 Placement Optimizer
**What to do:**
Calculate optimal placement positions for cards

**How to do it:**
```python
# src/recommendation/placement_optimizer.py
from typing import Tuple, List
from src.detection.game_state import GameState, Card, Unit

class PlacementOptimizer:
    def __init__(self):
        self.grid_size = (18, 32)
        
        # Define strategic positions
        self.key_positions = {
            'king_tower': (9, 30),
            'princess_left': (4, 25),
            'princess_right': (14, 25),
            'bridge_left': (4, 16),
            'bridge_right': (14, 16),
            'center': (9, 20)
        }
        
    def optimal_defensive_placement(
        self,
        threat_unit: Unit,
        counter_card: Card,
        state: GameState
    ) -> Tuple[int, int]:
        """
        Calculate optimal defensive placement
        
        Strategy:
        - Melee units: Place in front of threat
        - Ranged units: Place behind threat
        - Spells: Place on threat position
        - Buildings: Place in center or near tower
        """
        threat_pos = threat_unit.grid_position
        card_type = self._get_card_type(counter_card.name)
        
        if card_type == 'spell':
            # Place spell directly on threat
            return threat_pos
            
        elif card_type == 'building':
            # Place building in defensive position
            return self._defensive_building_position(threat_pos)
            
        elif self._is_melee_card(counter_card.name):
            # Place melee unit to intercept
            return self._intercept_position(threat_pos)
            
        else:  # Ranged unit
            # Place behind threat or to the side
            return self._ranged_counter_position(threat_pos)
            
    def optimal_offensive_placement(
        self,
        card: Card,
        target_lane: str,
        state: GameState
    ) -> Tuple[int, int]:
        """
        Calculate optimal offensive placement
        
        Strategy:
        - Tanks: Place at bridge
        - Support units: Place behind tank
        - Win conditions: Place at bridge or behind tank
        """
        card_role = self._get_card_role(card.name)
        
        if target_lane == 'left':
            bridge_pos = self.key_positions['bridge_left']
        elif target_lane == 'right':
            bridge_pos = self.key_positions['bridge_right']
        else:  # center
            bridge_pos = self.key_positions['center']
            
        if card_role == 'tank':
            # Tanks go at bridge
            return bridge_pos
            
        elif card_role == 'support':
            # Support units behind bridge
            return (bridge_pos[0], bridge_pos[1] + 3)
            
        elif card_role == 'spell':
            # Spells target tower or enemy units
            return self._offensive_spell_position(target_lane, state)
            
        else:  # Win condition
            return bridge_pos
            
    def _defensive_building_position(self, threat_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate building placement to pull aggro"""
        threat_col, threat_row = threat_pos
        
        # Place building 4 tiles from river, center or offset
        if threat_col < 9:  # Left lane
            return (6, 20)
        elif threat_col > 9:  # Right lane
            return (12, 20)
        else:  # Center
            return (9, 20)
            
    def _intercept_position(self, threat_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate position to intercept threat"""
        col, row = threat_pos
        
        # Place 2-3 tiles in front of threat
        intercept_row = min(row + 3, 28)
        
        return (col, intercept_row)
        
    def _ranged_counter_position(self, threat_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate ranged unit counter position"""
        col, row = threat_pos
        
        # Place to the side and slightly back
        offset_col = max(0, min(col + 2, 17))
        offset_row = min(row + 2, 28)
        
        return (offset_col, offset_row)
        
    def _get_card_type(self, card_name: str) -> str:
        """Get card type: troop, spell, or building"""
        # Load from card database
        from src.utils.config import load_card_data
        card_data = load_card_data()
        return card_data[card_name]['type']
        
    def _get_card_role(self, card_name: str) -> str:
        """Get card role: tank, support, win_condition, etc."""
        # Define card roles
        roles = {
            'giant': 'tank',
            'knight': 'tank',
            'golem': 'tank',
            'musketeer': 'support',
            'wizard': 'support',
            'witch': 'support',
            'hog_rider': 'win_condition',
            'balloon': 'win_condition',
            'fireball': 'spell',
            'arrows': 'spell',
        }
        return roles.get(card_name, 'support')
        
    def _is_melee_card(self, card_name: str) -> bool:
        """Check if card is melee"""
        from src.utils.config import load_card_data
        card_data = load_card_data()
        return card_data[card_name].get('range', 'melee') == 'melee'
```

**Expected Output:** Smart placement calculation system

#### 4.4 Elixir Management
**What to do:**
Implement intelligent elixir usage decisions

**How to do it:**
```python
# src/recommendation/elixir_manager.py
from typing import Optional
from src.detection.game_state import GameState, Card

class ElixirManager:
    def __init__(self):
        self.elixir_threshold_defense = 3  # Minimum for emergency defense
        self.elixir_threshold_offense = 6  # Good amount for push
        
    def should_play_card(
        self,
        card: Card,
        state: GameState,
        is_defensive: bool
    ) -> bool:
        """
        Decide if card should be played based on elixir
        
        Args:
            card: Card to potentially play
            state: Current game state
            is_defensive: Whether this is a defensive play
            
        Returns:
            True if card should be played
        """
        elixir_after = state.elixir_current - card.elixir_cost
        
        # Always defend if necessary
        if is_defensive and state.enemy_units:
            # Allow going low on elixir for critical defense
            return elixir_after >= 0
            
        # For offensive plays, maintain elixir reserve
        if not is_defensive:
            # Don't overcommit
            return elixir_after >= self.elixir_threshold_defense
            
        return True
        
    def calculate_elixir_advantage(self, state: GameState) -> float:
        """
        Estimate elixir advantage/disadvantage
        
        Returns:
            Positive value = ahead on elixir
            Negative value = behind on elixir
        """
        # Estimate enemy elixir based on units deployed
        enemy_elixir_spent = sum(
            self._estimate_card_cost(unit.unit_type)
            for unit in state.enemy_units
        )
        
        friendly_elixir_spent = sum(
            self._estimate_card_cost(unit.unit_type)
            for unit in state.friendly_units
        )
        
        # Simple approximation
        elixir_advantage = friendly_elixir_spent - enemy_elixir_spent
        
        return elixir_advantage
        
    def _estimate_card_cost(self, card_name: str) -> int:
        """Estimate elixir cost of card"""
        from src.utils.config import load_card_data
        card_data = load_card_data()
        return card_data.get(card_name, {}).get('elixir_cost', 4)
        
    def prioritize_cards_by_elixir(
        self,
        cards: List[Card],
        available_elixir: int
    ) -> List[Card]:
        """
        Sort cards by elixir efficiency
        
        Returns:
            Cards sorted by priority given current elixir
        """
        affordable_cards = [c for c in cards if c.elixir_cost <= available_elixir]
        
        # Sort by cost (prefer cheaper cards when low on elixir)
        if available_elixir < 5:
            affordable_cards.sort(key=lambda c: c.elixir_cost)
        else:
            # When elixir is high, prefer higher cost cards
            affordable_cards.sort(key=lambda c: -c.elixir_cost)
            
        return affordable_cards
```

**Expected Output:** Elixir-aware recommendation system

### Deliverables
- [ ] Strategy framework implemented
- [ ] Counter card database complete
- [ ] Placement optimizer working
- [ ] Elixir management system
- [ ] Integration of all recommendation components

### Testing Criteria
- Recommendations make strategic sense
- Counter cards are appropriate for threats
- Placement positions are optimal
- Elixir is managed efficiently
- System handles all card types

---

## Phase 5: Integration & Real-time Processing

**Duration:** 3-4 days  
**Priority:** High  
**Assigned To:** All team members

### Objectives
- Integrate all components
- Implement real-time processing loop
- Connect detection to UI
- Optimize performance

### Tasks

#### 5.1 Main Processing Loop
**What to do:**
Create main loop that coordinates all systems

**How to do it:**
```python
# src/main.py
import tkinter as tk
from src.ui.main_window import TorchRoyaleApp
from src.capture.screen_capture import ScreenCapture
from src.detection.model_loader import ModelLoader
from src.detection.card_detector import CardDetector
from src.detection.board_analyzer import BoardAnalyzer
from src.detection.game_state import GameStateManager
from src.recommendation.strategy import StrategyManager
from src.recommendation.placement_optimizer import PlacementOptimizer
import threading
import time

class TorchRoyaleSystem:
    def __init__(self):
        # Initialize components
        self.model_loader = ModelLoader()
        self.model = self.model_loader.load_model()
        
        self.card_detector = CardDetector(self.model)
        self.board_analyzer = BoardAnalyzer(self.model)
        self.game_state_manager = GameStateManager(
            self.card_detector,
            self.board_analyzer
        )
        
        self.strategy_manager = StrategyManager('balanced')
        self.placement_optimizer = PlacementOptimizer()
        
        self.screen_capture = ScreenCapture(target_fps=2)
        
        # UI
        self.app = TorchRoyaleApp()
        self.app.set_detection_callback(self.toggle_detection)
        
        # State
        self.is_running = False
        self.detection_thread = None
        
    def start(self):
        """Start the application"""
        # Warm up model
        print("Warming up model...")
        self.model_loader.warm_up()
        print("Ready!")
        
        # Start UI
        self.app.run()
        
    def toggle_detection(self):
        """Start/stop detection"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()
            
    def start_detection(self):
        """Start detection thread"""
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.app.update_status("Detection running...")
        
    def stop_detection(self):
        """Stop detection thread"""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
            
        self.app.update_status("Detection stopped")
        
    def _detection_loop(self):
        """Main detection and recommendation loop"""
        while self.is_running:
            try:
                # Capture screenshot
                screenshot = self.screen_capture.capture()
                if screenshot is None:
                    time.sleep(0.01)
                    continue
                    
                # Update game state
                state = self.game_state_manager.update_state(screenshot)
                
                # Validate state
                if not self._validate_state(state):
                    continue
                    
                # Get recommendation
                recommendation = self.strategy_manager.get_recommendation(state)
                
                if recommendation:
                    card, position = recommendation
                    
                    # Display recommendation in UI
                    self.app.show_recommendation(
                        card_name=card.name,
                        grid_position=position,
                        confidence=card.confidence
                    )
                    
                    # Log recommendation
                    self._log_recommendation(card, position, state)
                    
                # Update UI with game state
                self.app.update_game_state(state)
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
                
    def _validate_state(self, state) -> bool:
        """Validate game state"""
        # Basic validation
        if not state.hand_cards:
            return False
        if state.confidence_score < 0.5:
            return False
        return True
        
    def _log_recommendation(self, card, position, state):
        """Log recommendation for analysis"""
        from src.utils.logger import log_recommendation
        log_recommendation({
            'card': card.name,
            'position': position,
            'elixir': state.elixir_current,
            'enemy_units': len(state.enemy_units),
            'timestamp': state.timestamp
        })

if __name__ == "__main__":
    system = TorchRoyaleSystem()
    system.start()
```

**Expected Output:** Fully integrated system

#### 5.2 UI Integration
**What to do:**
Connect detection system to UI updates

**How to do it:**
```python
# Extension to src/ui/main_window.py
def set_detection_callback(self, callback):
    """Set callback for detection start/stop"""
    self.detection_callback = callback
    
def _toggle_detection(self):
    """Toggle detection on/off"""
    if self.detection_callback:
        self.detection_callback()
        
def show_recommendation(self, card_name: str, grid_position: tuple, confidence: float):
    """Display recommendation on board"""
    # Update on main thread
    self.root.after(0, lambda: self._display_recommendation(card_name, grid_position, confidence))
    
def _display_recommendation(self, card_name, grid_position, confidence):
    """Actually display recommendation (called on main thread)"""
    self.card_display.show_recommendation(card_name, grid_position, confidence)
    
def update_game_state(self, state):
    """Update UI with current game state"""
    self.root.after(0, lambda: self._update_state_display(state))
    
def _update_state_display(self, state):
    """Update state display (called on main thread)"""
    # Update elixir display
    elixir_text = f"Elixir: {state.elixir_current}/{state.elixir_max}"
    self.elixir_label.config(text=elixir_text)
    
    # Update hand cards display
    self._display_hand_cards(state.hand_cards)
    
    # Update board units
    self._display_board_units(state.friendly_units, state.enemy_units)
```

**Expected Output:** Smooth UI updates

#### 5.3 Performance Optimization
**What to do:**
Optimize system for real-time performance

**How to do it:**
```python
# src/utils/performance.py
import time
import functools
from typing import Callable

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        
    def measure(self, name: str):
        """Decorator to measure function execution time"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(elapsed)
                
                # Keep only recent measurements
                if len(self.timings[name]) > 100:
                    self.timings[name] = self.timings[name][-100:]
                    
                return result
            return wrapper
        return decorator
        
    def get_average_time(self, name: str) -> float:
        """Get average execution time for function"""
        if name not in self.timings:
            return 0.0
        return sum(self.timings[name]) / len(self.timings[name])
        
    def print_report(self):
        """Print performance report"""
        print("\n=== Performance Report ===")
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            print(f"{name}: {avg_time*1000:.2f}ms (avg of {len(times)} calls)")

# Usage
perf_monitor = PerformanceMonitor()

@perf_monitor.measure("card_detection")
def detect_cards(image):
    # ... detection code
    pass
```

**Optimization Targets:**
- Total loop time: <500ms
- Model inference: <300ms
- State update: <50ms
- UI update: <50ms
- Recommendation generation: <100ms

**Expected Output:** Performance monitoring and optimization

### Deliverables
- [ ] Main processing loop implemented
- [ ] UI integration complete
- [ ] Performance optimized
- [ ] Thread safety ensured
- [ ] Error handling robust

### Testing Criteria
- System runs continuously without crashes
- UI updates smoothly
- Recommendations appear within 1 second
- Performance targets met
- No memory leaks

---

## Phase 6: Testing & Optimization

**Duration:** 3-4 days  
**Priority:** High  
**Assigned To:** All team members

### Objectives
- Comprehensive testing
- Bug fixing
- Performance tuning
- User experience refinement

### Tasks

#### 6.1 Unit Testing
**What to do:**
Create comprehensive unit tests

**How to do it:**
```python
# tests/test_detection.py
import pytest
from src.detection.card_detector import CardDetector
from src.detection.board_analyzer import BoardAnalyzer

class TestCardDetection:
    @pytest.fixture
    def detector(self):
        from src.detection.model_loader import ModelLoader
        model_loader = ModelLoader()
        return CardDetector(model_loader.load_model())
        
    def test_hand_card_detection(self, detector):
        """Test detecting cards in hand"""
        test_image = load_test_image('test_hand_cards.jpg')
        cards = detector.detect_hand_cards(test_image)
        
        assert len(cards) == 4, "Should detect 4 cards in hand"
        assert all(c.confidence > 0.5 for c in cards), "All cards should have >50% confidence"
        
    def test_elixir_detection(self, detector):
        """Test elixir counter detection"""
        test_image = load_test_image('test_elixir.jpg')
        current, max_elixir = detector.detect_elixir(test_image)
        
        assert 0 <= current <= 10, "Elixir should be 0-10"
        assert max_elixir == 10, "Max elixir should be 10"

# tests/test_recommendation.py
class TestRecommendation:
    def test_counter_selection(self):
        """Test counter card selection"""
        # Create mock game state with giant
        # Should recommend mini pekka, inferno tower, or skeleton army
        pass
        
    def test_placement_optimization(self):
        """Test placement position calculation"""
        # Verify positions are within valid grid
        # Verify positions make strategic sense
        pass
```

**Expected Output:** Comprehensive test suite with >80% coverage

#### 6.2 Integration Testing
**What to do:**
Test complete system end-to-end

**Test Scenarios:**
1. Start application, begin detection, get recommendation
2. Handle enemy push (multiple units)
3. Build offensive push
4. Respond to spell
5. Manage low elixir situation
6. Handle detection failures gracefully

**How to do it:**
```python
# tests/test_integration.py
import pytest
from src.main import TorchRoyaleSystem

class TestIntegration:
    @pytest.fixture
    def system(self):
        return TorchRoyaleSystem()
        
    def test_full_cycle(self, system):
        """Test complete detection-recommendation-display cycle"""
        # Load test screenshot
        test_screenshot = load_test_image('full_game_state.jpg')
        
        # Process through full pipeline
        state = system.game_state_manager.update_state(test_screenshot)
        recommendation = system.strategy_manager.get_recommendation(state)
        
        assert state is not None, "Should generate game state"
        assert recommendation is not None, "Should generate recommendation"
        
        card, position = recommendation
        assert card in state.hand_cards, "Recommended card should be in hand"
        assert 0 <= position[0] < 18 and 0 <= position[1] < 32, "Position should be valid"
```

**Expected Output:** All integration tests passing

#### 6.3 User Testing
**What to do:**
Test with actual Clash Royale gameplay

**Test Plan:**
1. Set up application alongside Clash Royale game
2. Run training mode battles (not ranked)
3. Follow recommendations manually
4. Record accuracy and usefulness of recommendations
5. Identify edge cases and failures

**Metrics to Track:**
- Detection accuracy (% of correct detections)
- Recommendation quality (subjective rating)
- Response time (time from action to recommendation)
- False positives/negatives
- User satisfaction

**Expected Output:** User testing report with findings

#### 6.4 Bug Fixing
**What to do:**
Fix all identified bugs and issues

**Common Issues to Address:**
- Detection failures in certain arenas
- Incorrect card identification
- Poor recommendations in specific situations
- UI freezing or lag
- Memory leaks from continuous operation
- Thread safety issues

**Expected Output:** Stable, bug-free application

#### 6.5 Performance Tuning
**What to do:**
Optimize for best performance

**Optimization Areas:**
1. Model inference (try quantization, pruning)
2. Image preprocessing (reduce resolution if acceptable)
3. Caching (cache recent detections)
4. Thread management (optimize thread sleep times)
5. UI rendering (reduce update frequency if needed)

**Expected Output:** Optimized system meeting performance targets

### Deliverables
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] User testing completed
- [ ] All critical bugs fixed
- [ ] Performance optimized

### Testing Criteria
- Test coverage >80%
- All tests passing
- User testing shows >85% satisfaction
- No critical or high-priority bugs
- Performance targets met

---

## Phase 7: Documentation & Presentation

**Duration:** 2-3 days  
**Priority:** Medium  
**Assigned To:** All team members

### Objectives
- Complete documentation
- Create user guide
- Prepare presentation
- Record demo video

### Tasks

#### 7.1 Code Documentation
**What to do:**
Document all code with docstrings and comments

**How to do it:**
```python
# Example of proper documentation
def detect_hand_cards(self, image: np.ndarray) -> List[Card]:
    """
    Detect cards in player's hand from screenshot.
    
    This function analyzes the bottom portion of the game screenshot
    to identify the 4 cards currently available to the player. It uses
    the Roboflow model to detect card types and positions.
    
    Args:
        image: numpy array of game screenshot in RGB format
               Expected shape: (height, width, 3)
               Expected size: 1920x1080 or similar
               
    Returns:
        List of Card objects, sorted by position (left to right)
        Each Card contains:
            - name: str (e.g., "knight", "fireball")
            - elixir_cost: int (1-10)
            - position_index: int (0-3)
            - confidence: float (0-1)
            
    Raises:
        ValueError: If image is invalid format or size
        ModelError: If Roboflow model inference fails
        
    Example:
        >>> detector = CardDetector(model)
        >>> screenshot = capture_screen()
        >>> cards = detector.detect_hand_cards(screenshot)
        >>> for card in cards:
        ...     print(f"{card.name} at position {card.position_index}")
        
    Notes:
        - Confidence threshold is set to 60% for hand detection
        - Cards are automatically sorted by x-position
        - If fewer than 4 cards detected, check image quality
    """
    # Implementation...
```

**Expected Output:** All functions and classes documented

#### 7.2 User Guide
**What to do:**
Create comprehensive user guide

**User Guide Sections:**
1. Installation
   - System requirements
   - Python setup
   - Dependency installation
   - Roboflow API setup
   
2. Configuration
   - Window positioning
   - Strategy selection
   - Performance settings
   
3. Usage
   - Starting the application
   - Beginning detection
   - Interpreting recommendations
   - Manual card placement
   
4. Troubleshooting
   - Common issues
   - Error messages
   - Performance problems
   
5. Fair Play
   - Usage guidelines
   - Compliance with game policies
   - Ethical considerations

**How to do it:**
Create `docs/user_guide.md` with step-by-step instructions and screenshots

**Expected Output:** Complete user guide

#### 7.3 Technical Documentation
**What to do:**
Document system architecture and design decisions

**Technical Docs Sections:**
1. System Architecture
   - Component diagram
   - Data flow
   - Thread model
   
2. Detection System
   - Model details
   - Preprocessing pipeline
   - Accuracy metrics
   
3. Recommendation Algorithm
   - Strategy framework
   - Counter card logic
   - Placement optimization
   
4. Performance
   - Benchmarks
   - Optimization techniques
   - Resource usage
   
5. Future Enhancements
   - Potential improvements
   - Known limitations
   - Extension points

**Expected Output:** Technical documentation

#### 7.4 Presentation Preparation
**What to do:**
Create presentation slides for final project presentation

**Presentation Outline:**
1. Introduction (2 min)
   - Team introduction
   - Project overview
   - Problem statement
   
2. Technical Approach (3 min)
   - Architecture overview
   - Key technologies
   - Design decisions
   
3. Demo (5 min)
   - Live demonstration
   - Key features showcase
   - Recommendation examples
   
4. Results (2 min)
   - Performance metrics
   - Accuracy results
   - User feedback
   
5. Challenges & Solutions (2 min)
   - Technical challenges
   - How they were solved
   - Lessons learned
   
6. Conclusion (1 min)
   - Project success
   - Future work
   - Q&A

**Expected Output:** Presentation slides

#### 7.5 Demo Video
**What to do:**
Record demonstration video

**Video Sections:**
1. Introduction (30 sec)
2. Setup and launch (1 min)
3. Detection demonstration (2 min)
4. Recommendation showcase (2 min)
5. Different scenarios (2 min)
6. Conclusion (30 sec)

**Expected Output:** Demo video

### Deliverables
- [ ] Code fully documented
- [ ] User guide complete
- [ ] Technical documentation complete
- [ ] Presentation slides ready
- [ ] Demo video recorded
- [ ] Project report written

### Testing Criteria
- Documentation is clear and complete
- New users can follow user guide successfully
- Presentation meets time requirements
- Demo video shows all key features

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     TorchRoyale System                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Screen     │────────▶│   Detection  │                 │
│  │   Capture    │         │   Pipeline   │                 │
│  └──────────────┘         └──────┬───────┘                 │
│                                   │                          │
│                                   ▼                          │
│                          ┌──────────────┐                   │
│                          │  Game State  │                   │
│                          │   Manager    │                   │
│                          └──────┬───────┘                   │
│                                   │                          │
│                                   ▼                          │
│                          ┌──────────────┐                   │
│                          │ Recommendation│                  │
│                          │    Engine    │                   │
│                          └──────┬───────┘                   │
│                                   │                          │
│                                   ▼                          │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Tkinter    │◀────────│  Display     │                 │
│  │      UI      │         │  Controller  │                 │
│  └──────────────┘         └──────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Capture**: Screenshot taken at 2 FPS
2. **Detection**: Roboflow model detects cards and units
3. **State Update**: Game state object updated
4. **Strategy**: Current strategy analyzes state
5. **Recommendation**: Best card and position determined
6. **Display**: UI updated with recommendation

### Threading Model

- **Main Thread**: Tkinter UI
- **Detection Thread**: Screen capture and processing
- **Communication**: Thread-safe queues and callbacks

---

## Risk Mitigation Strategies

### Risk 1: Model Accuracy Issues
**Mitigation:**
- Extensive testing before integration
- Fine-tuning capability built in
- Confidence thresholds
- Manual override options

### Risk 2: Game Updates
**Mitigation:**
- Modular detection layer
- Version-locked model
- Quick retraining capability
- Fallback to previous version

### Risk 3: Performance Issues
**Mitigation:**
- Performance monitoring built in
- Optimization passes scheduled
- Configurable quality settings
- Hardware requirements documented

### Risk 4: Fair Play Compliance
**Mitigation:**
- No game automation
- Manual card placement required
- Clear usage guidelines
- Training mode only

---

## Project Milestones

| Milestone | Target Date | Deliverables |
|-----------|------------|--------------|
| Phase 0 Complete | Day 2 | Environment setup done |
| Phase 1 Complete | Day 7 | Model integrated and tested |
| Phase 2 Complete | Day 11 | UI foundation ready |
| Phase 3 Complete | Day 16 | Detection system working |
| Phase 4 Complete | Day 23 | Recommendation algorithm complete |
| Phase 5 Complete | Day 27 | Full system integrated |
| Phase 6 Complete | Day 31 | Testing and optimization done |
| Phase 7 Complete | Day 34 | Documentation and presentation ready |
| **Final Presentation** | **Day 35** | **Project complete** |

---

## Success Metrics

### Technical Metrics
- Detection accuracy: >90%
- Recommendation response time: <1 second
- System uptime: >95% (no crashes)
- Memory usage: <2GB
- CPU usage: <50% average

### User Experience Metrics
- User satisfaction: >85%
- Recommendation quality rating: >4/5
- Ease of use rating: >4/5
- Would recommend to others: >80%

### Project Metrics
- Code coverage: >80%
- Documentation completeness: 100%
- On-time delivery: Yes
- Team collaboration score: >4/5

---

## Conclusion

This comprehensive project plan provides a detailed roadmap for building TorchRoyale from setup through completion. Each phase builds on previous work, with clear objectives, tasks, deliverables, and success criteria.

The plan balances technical ambition with practical execution, allowing for iterative development and testing. Regular checkpoints ensure the project stays on track.

**Key Success Factors:**
1. Strong team collaboration
2. Regular testing and validation
3. Focus on user experience
4. Clear documentation
5. Ethical compliance

With careful execution of this plan, TorchRoyale will achieve its goal of providing helpful card recommendations while maintaining fair play principles.

---

**Document Version:** 1.0  
**Last Updated:** January 13, 2026  
**Next Review:** End of Phase 1

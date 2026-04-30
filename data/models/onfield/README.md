# On-Field Detection Models

This directory contains the trained YOLOv8 models used by `DualModelDetector`
for detecting cards on the battlefield.

## Models

### cicadas_best.pt
- **Purpose**: Detects PLAYER'S cards on the field
- **Dataset**: Cicadas dataset (Roboflow workspace: `cicadas`, project: `clash-royale-9eug2`)
- **Cards detected**: Hog 2.6 deck -- cannon, fireball, hog_rider, ice_golem, ice_spirit, log, musketeer, skeletons, evo_ice_spirit, evo_skeletons
- **Input**: YOLOv8n trained on arena crop images (576x896)

### visionbot_best.pt
- **Purpose**: Detects OPPONENT'S cards on the field
- **Dataset**: Vision Bot all-enemy-cards dataset (Roboflow workspace: `vision-bot`, project: `clash-royale-all-enemy-cards-w9haz`)
- **Cards detected**: All cards in the game (comprehensive opponent card coverage)
- **Input**: YOLOv8n trained on arena crop images (576x896)

## Regenerating the Models

To retrain or regenerate either model:

1. Download the datasets:
   ```bash
   python scripts/download_datasets.py
   ```

2. Train the Cicadas model (player cards):
   ```bash
   python scripts/train_cicadas.py
   ```

3. Train the Vision Bot model (opponent cards):
   ```bash
   python scripts/train_visionbot.py
   ```

Both scripts will automatically:
- Verify dataset existence (error clearly if missing)
- Detect the best available device (MPS/CUDA/CPU)
- Train YOLOv8n for 100 epochs at 640px resolution
- Copy the best weights to the canonical paths after training

## Notes

- The model assignment (Cicadas vs Vision Bot) determines ownership -- not tile position.
- Cross-model NMS is applied where both models detect overlapping boxes.
- See `src/detection/dual_model_detector.py` for the full detection pipeline.

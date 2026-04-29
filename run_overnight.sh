#!/bin/bash
set -e
mkdir -p logs

echo "=== [1/4] Patch OCR fields (timer, elixir, multiplier, phase) ===" | tee logs/patch.log
venv/bin/python3 scripts/patch_ocr_fields.py \
    --analyses-dir output/analysis \
    --preload-ocr \
    2>&1 | tee -a logs/patch.log

echo "=== [2/4] Patch hand_cards via HandTracker ===" | tee logs/hand.log
venv/bin/python3 scripts/patch_hand_cards.py \
    --analyses-dir output/analysis \
    2>&1 | tee -a logs/hand.log

echo "=== [3/4] Convert to episodes and merge ===" | tee logs/convert.log
venv/bin/python3 scripts/convert_to_episodes.py output/analysis \
    --output output/pkl/new_episodes.pkl \
    2>&1 | tee -a logs/convert.log

venv/bin/python3 -c "
import pickle
old = pickle.load(open('/Volumes/SanDisk128G/torchroyale-data/output/pkl/all_combined.pkl','rb'))
new = pickle.load(open('output/pkl/new_episodes.pkl','rb'))
combined = old + new
pickle.dump(combined, open('output/pkl/all_combined.pkl','wb'))
print(f'Merged: {len(old)} existing + {len(new)} new = {len(combined)} total')
" 2>&1 | tee -a logs/convert.log

echo "=== [4/4] Train 400 epochs on MPS ===" | tee logs/train.log
venv/bin/python3 -m src.transformer.train \
    --episodes output/pkl/all_combined.pkl \
    --output data/models/dt \
    --epochs 400 \
    --device mps \
    2>&1 | tee -a logs/train.log

echo "=== Done ===" | tee -a logs/train.log

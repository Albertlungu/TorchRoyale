#!/bin/bash
set -e

mkdir -p logs

echo "=== [1/4] Fixing on-field classifications ===" | tee logs/fix_onfield.log
venv/bin/python3 scripts/fix_onfield_classifications.py \
    "output/analysis/Game 1_analysis.json" \
    "output/analysis/Game 2_analysis.json" \
    "output/analysis/Game 3_analysis.json" \
    "output/analysis/Game 4_analysis.json" \
    "output/analysis/Game 5_analysis.json" \
    "output/analysis/Game 6_analysis.json" \
    "output/analysis/Game 7_analysis.json" \
    "output/analysis/Game 8_analysis.json" \
    "output/analysis/Game 9_analysis.json" \
    2>&1 | tee -a logs/fix_onfield.log

echo "=== [2/4] Converting Game JSONs to episodes ===" | tee logs/convert.log
venv/bin/python3 scripts/convert_json_to_episodes.py \
    "output/analysis/Game 1_analysis.json" \
    "output/analysis/Game 2_analysis.json" \
    "output/analysis/Game 3_analysis.json" \
    "output/analysis/Game 4_analysis.json" \
    "output/analysis/Game 5_analysis.json" \
    "output/analysis/Game 6_analysis.json" \
    "output/analysis/Game 7_analysis.json" \
    "output/analysis/Game 8_analysis.json" \
    "output/analysis/Game 9_analysis.json" \
    --output output/pkl/game_episodes_new.pkl \
    2>&1 | tee -a logs/convert.log

echo "=== [3/4] Merging with existing 13 episodes ===" | tee logs/merge.log
venv/bin/python3 -c "
import pickle
old = pickle.load(open('output/pkl/all_episodes.pkl', 'rb'))
new = pickle.load(open('output/pkl/game_episodes_new.pkl', 'rb'))
combined = old + new
pickle.dump(combined, open('output/pkl/all_combined.pkl', 'wb'))
print(f'Merged: {len(old)} existing + {len(new)} new = {len(combined)} total episodes')
" 2>&1 | tee -a logs/merge.log

echo "=== [4/4] Training for 400 epochs on MPS ===" | tee logs/train.log
venv/bin/python3 -m src.transformer.train \
    --episodes output/pkl/all_combined.pkl \
    --output data/models/dt \
    --epochs 400 \
    --device mps \
    2>&1 | tee -a logs/train.log

echo "=== Done ===" | tee -a logs/train.log

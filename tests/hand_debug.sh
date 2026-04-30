#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import cv2, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
# Use canonical hand classifier geometry for box positions
from src.detection.hand_classifier import (
    _Y_TOP_FRAC,
    _Y_BOT_FRAC,
    _INSET_FRAC,
    _SLOT_OFFSET_FRACS,
    get_next_bbox,
)


video = "data/replays/Game_23.mp4"
frame_no = 2800
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ret, frame = cap.read()
cap.release()
if not ret:
    print("Failed to read frame", frame_no); sys.exit(1)

h,w = frame.shape[:2]
# detect game strip columns like HandClassifier
gray = np.mean(frame, axis=2)
cols = np.where(np.mean(gray, axis=0) > 30)[0]
if cols.size:
    x_left, x_right = int(cols.min()), int(cols.max())
else:
    x_left, x_right = 0, w

game_w = x_right - x_left
# Hand vertical bounds
y_top = int(h * _Y_TOP_FRAC)
y_bot = int(h * _Y_BOT_FRAC)
# Next-card pixel bbox using helper
nx1, ny1, nx2, ny2 = get_next_bbox(h, w, x_left, x_right)
next_end = nx2
cards_w = x_right - next_end
slot_w = cards_w // 4 if cards_w > 0 else w
inset = int(slot_w * _INSET_FRAC)

# Draw the next-card preview region (left of slot 0) with distinct styling
if next_end > x_left:
    nx1, ny1, nx2, ny2 = get_next_bbox(h, w, x_left, x_right)
    # outer red thick border to distinguish the Next preview
    cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 0, 255), 3)
    # inner inset to visualise the preview content area
    nx_w = max(1, nx2 - nx1)
    inset_x = int(nx_w * 0.12)
    inset_y = int((ny2 - ny1) * 0.08)
    ix1 = nx1 + inset_x
    iy1 = ny1 + inset_y
    ix2 = nx2 - inset_x
    iy2 = ny2 - inset_y
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 255), 1)
    cv2.putText(frame, "Next Card", (nx1 + 6, max(0, ny1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

for i, off_frac in enumerate(_SLOT_OFFSET_FRACS):
    off = int(slot_w * off_frac)
    x1 = max(0, next_end + i * slot_w + inset + off)
    x2 = min(w, x1 + slot_w - 2 * inset)
    cv2.rectangle(frame, (x1, y_top), (x2, y_bot), (0, 255, 0), 2)

cv2.imshow("Hand boxes", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Displayed hand debug window (press any key to close)")
PY

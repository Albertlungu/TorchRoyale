"""
Two-stage behavioral cloning model trainer.

Stage 1 - Card Selection:
    Predicts which hand card to play (index 0-3).

Stage 2 - Position Selection:
    Predicts where to place the chosen card (tile_y * 18 + tile_x).

Both stages use scikit-learn RandomForestClassifier, saved with joblib.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.feature_encoder import encode, FEATURE_DIM, VOCAB_SIZE, _card_name_to_id


DEFAULT_MODEL_DIR = Path(project_root) / "data" / "models"
STAGE1_FILENAME = "stage1_card.pkl"
STAGE2_FILENAME = "stage2_pos.pkl"


def _prepare_dataset(
    training_data: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw training data into feature matrix and label arrays.

    Args:
        training_data: List of {"state": ..., "action": ...} dicts.

    Returns:
        (X, y_card, y_pos) where:
            X: (N, FEATURE_DIM) feature matrix
            y_card: (N,) card index in hand (0-3), or -1 if not found
            y_pos: (N,) flattened tile position (tile_y * 18 + tile_x)
    """
    X_list = []
    y_card_list = []
    y_pos_list = []

    for example in training_data:
        state = example["state"]
        action = example["action"]

        features = encode(state)

        # Determine card index in hand
        played_card = action["card_name"]
        hand_cards = state.get("hand_cards", [])

        card_idx = -1
        played_clean = played_card.lower()
        for i, hc in enumerate(hand_cards):
            if hc.lower() == played_clean:
                card_idx = i
                break

        # Skip examples where the played card isn't found in hand
        # (detection noise - can't create a reliable label)
        if card_idx == -1:
            continue

        tile_x = action["tile_x"]
        tile_y = action["tile_y"]
        pos_label = tile_y * 18 + tile_x

        X_list.append(features)
        y_card_list.append(card_idx)
        y_pos_list.append(pos_label)

    if not X_list:
        return np.empty((0, FEATURE_DIM)), np.empty(0, dtype=int), np.empty(0, dtype=int)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_card_list, dtype=int),
        np.array(y_pos_list, dtype=int),
    )


def train(
    training_data_path: str = "data/training_data.json",
    model_dir: Optional[str] = None,
    n_estimators: int = 200,
    test_size: float = 0.2,
    verbose: bool = True,
) -> Tuple[RandomForestClassifier, RandomForestClassifier]:
    """
    Train the two-stage behavioral cloning model.

    Args:
        training_data_path: Path to the JSON training data.
        model_dir: Directory to save trained models. Defaults to data/models/.
        n_estimators: Number of trees in each Random Forest.
        test_size: Fraction of data held out for validation.
        verbose: Print training stats.

    Returns:
        (stage1_model, stage2_model) tuple of trained classifiers.
    """
    out_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    with open(training_data_path, "r") as f:
        raw_data = json.load(f)

    if verbose:
        print(f"Loaded {len(raw_data)} raw training examples.")

    X, y_card, y_pos = _prepare_dataset(raw_data)

    if len(X) == 0:
        raise ValueError(
            "No usable training examples (played cards were never found in hand). "
            "Check that the detection model is correctly identifying hand cards."
        )

    if verbose:
        print(f"Usable training examples: {len(X)}")
        print(f"Feature dimension: {FEATURE_DIM}")
        unique_cards = np.unique(y_card)
        unique_pos = np.unique(y_pos)
        print(f"Unique card indices: {unique_cards}")
        print(f"Unique positions: {len(unique_pos)}")

    # Split
    X_train, X_val, y_card_train, y_card_val, y_pos_train, y_pos_val = (
        train_test_split(X, y_card, y_pos, test_size=test_size, random_state=42)
    )

    # --- Stage 1: Card Selection ---
    if verbose:
        print("\n--- Stage 1: Card Selection ---")

    stage1 = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    stage1.fit(X_train, y_card_train)

    if len(X_val) > 0:
        card_pred = stage1.predict(X_val)
        card_acc = accuracy_score(y_card_val, card_pred)
        if verbose:
            print(f"Validation accuracy: {card_acc:.3f} (random baseline: {1/max(len(np.unique(y_card)), 1):.3f})")

    # --- Stage 2: Position Selection ---
    if verbose:
        print("\n--- Stage 2: Position Selection ---")

    # Augment features with one-hot of the chosen card for stage 2
    card_onehot_train = np.zeros((len(X_train), 4), dtype=np.float32)
    for i, c in enumerate(y_card_train):
        card_onehot_train[i, c] = 1.0
    X_train_s2 = np.hstack([X_train, card_onehot_train])

    stage2 = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    stage2.fit(X_train_s2, y_pos_train)

    if len(X_val) > 0:
        card_onehot_val = np.zeros((len(X_val), 4), dtype=np.float32)
        for i, c in enumerate(y_card_val):
            card_onehot_val[i, c] = 1.0
        X_val_s2 = np.hstack([X_val, card_onehot_val])
        pos_pred = stage2.predict(X_val_s2)
        pos_acc = accuracy_score(y_pos_val, pos_pred)
        if verbose:
            print(f"Validation accuracy: {pos_acc:.3f}")

    # Save models
    stage1_path = out_dir / STAGE1_FILENAME
    stage2_path = out_dir / STAGE2_FILENAME
    joblib.dump(stage1, stage1_path)
    joblib.dump(stage2, stage2_path)

    if verbose:
        print(f"\nModels saved to: {out_dir}")
        print(f"  Stage 1: {stage1_path}")
        print(f"  Stage 2: {stage2_path}")

    return stage1, stage2


def main():
    """CLI entry point for model training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the behavioral cloning card recommendation model."
    )
    parser.add_argument(
        "--data",
        default="data/training_data.json",
        help="Path to training data JSON (default: data/training_data.json)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to save models (default: data/models/)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees per Random Forest (default: 200)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()
    train(
        training_data_path=args.data,
        model_dir=args.model_dir,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

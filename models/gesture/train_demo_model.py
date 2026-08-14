"""
Train a small LSTM demo model on ~25 everyday signs.

Uses:
  - WLASL sequences: dataset/sign_language/wlasl_features/sequences/
  - Your recordings: dataset/sign_language/demo_features/sequences/

Run from project root:
    .\\tf_venv\\Scripts\\python.exe models/gesture/train_demo_model.py

Outputs:
  - models/gesture/lstm_gesture_model.keras
  - models/gesture/labels.txt
  - models/gesture/demo_labels_guide.txt  (human-readable list)
"""

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gesture.demo_vocabulary import DEMO_WORDS, SIGN_GUIDE, CUSTOM_DEMO_WORDS, WLASL_DEMO_WORDS
from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

SEQUENCE_LENGTH = 30
FEATURES = 225
WLASL_SEQ = PROJECT_ROOT / "dataset" / "sign_language" / "wlasl_features" / "sequences"
CUSTOM_SEQ = PROJECT_ROOT / "dataset" / "sign_language" / "demo_features" / "sequences"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture" / "lstm_gesture_model.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "gesture" / "labels.txt"
GUIDE_PATH = PROJECT_ROOT / "models" / "gesture" / "demo_labels_guide.txt"


def sequence_dir_for_word(word: str) -> Path | None:
    if word in WLASL_DEMO_WORDS:
        p = WLASL_SEQ / word
        return p if p.is_dir() else None
    if word in CUSTOM_DEMO_WORDS:
        p = CUSTOM_SEQ / word
        return p if p.is_dir() else None
    return None


def load_sequences_for_word(word: str):
    class_dir = sequence_dir_for_word(word)
    if class_dir is None:
        return []

    samples = []
    for npy_file in class_dir.glob("*.npy"):
        res = np.load(npy_file)
        if res.shape == (SEQUENCE_LENGTH, FEATURES):
            samples.append(res)
        elif res.ndim == 2 and res.shape[1] == FEATURES:
            if res.shape[0] < SEQUENCE_LENGTH:
                pad = SEQUENCE_LENGTH - res.shape[0]
                res = np.pad(res, ((0, pad), (0, 0)), mode="constant")
            else:
                res = res[:SEQUENCE_LENGTH, :]
            samples.append(res.astype(np.float32))
    return samples


def load_demo_data():
    labels = []
    X, y = [], []

    print("\nLoading demo vocabulary:")
    print("-" * 50)

    for word in DEMO_WORDS:
        samples = load_sequences_for_word(word)
        if not samples:
            print(f"  [SKIP] {word:12} — no sequences (record or check dataset)")
            continue

        labels.append(word)
        class_id = len(labels) - 1
        for seq in samples:
            X.append(seq)
            y.append(class_id)
        source = "WLASL" if word in WLASL_DEMO_WORDS else "YOU"
        print(f"  [OK]   {word:12} — {len(samples):3} samples  ({source})")

    print("-" * 50)
    print(f"Total: {len(X)} sequences, {len(labels)} classes\n")

    if not X:
        raise RuntimeError("No training data found. See docs/DEMO_WORDS.md")

    missing_custom = [w for w in CUSTOM_DEMO_WORDS if w not in labels]
    if missing_custom:
        print("Warning: custom words not recorded yet:", ", ".join(missing_custom))
        print("Run: python models/gesture/collect_demo_sequences.py\n")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels


def build_model(num_classes: int):
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQUENCE_LENGTH, FEATURES)),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False, activation="relu"),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def write_guide(labels):
    lines = [
        "SignVision AI — Demo vocabulary",
        "=" * 40,
        f"Trained words: {len(labels)}",
        "",
        "Index | Word      | How to sign (short guide)",
        "-" * 40,
    ]
    for i, word in enumerate(labels):
        guide = SIGN_GUIDE.get(word, "Perform the ASL sign clearly for 1–2 seconds.")
        lines.append(f"{i+1:5} | {word:9} | {guide}")

    lines += [
        "",
        "Tips for live demo:",
        "  • Face the webcam with good lighting.",
        "  • Hold each sign steady for about 1–2 seconds.",
        "  • Keep both hands visible when possible.",
        "  • Run the web app: python web_app.py",
        "",
        "WLASL words were trained from pre-extracted sequences.",
        "Custom words (hello, hi, welcome, thanks, please) need your recordings.",
    ]
    GUIDE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Guide written to {GUIDE_PATH}")


def train():
    X, y, labels = load_demo_data()

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = max(1, int(len(X) * 0.85))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(num_classes=len(labels))
    model.summary()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    print("Training demo model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    LABELS_PATH.write_text("\n".join(labels) + "\n", encoding="utf-8")
    write_guide(labels)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Labels saved: {LABELS_PATH}")
    print("\nTrained words:")
    for i, w in enumerate(labels, 1):
        print(f"  {i:2}. {w}")


if __name__ == "__main__":
    train()

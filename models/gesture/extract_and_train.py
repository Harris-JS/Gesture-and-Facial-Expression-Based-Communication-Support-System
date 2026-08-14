"""
extract_and_train.py
====================
Extracts MediaPipe hand landmarks from the local ASL image dataset
(dataset/dataset/ASL Dataset/asl_dataset/<class>/*.jpeg) and trains
a lightweight Dense (static-frame) classifier.

Run from the project root:
    python models/gesture/extract_and_train.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
import cv2
import mediapipe as mp

from utils.runtime_env import configure_runtime_environment
configure_runtime_environment()

import tensorflow as tf

# ── Config ───────────────────────────────────────────────────────────────────

# Primary dataset path (asl_dataset sub-folder has a-z)
DATASET_PATHS = [
    PROJECT_ROOT / "dataset" / "dataset" / "ASL Dataset" / "asl_dataset",
    PROJECT_ROOT / "dataset" / "dataset" / "ASL Dataset",
]

CSV_OUT       = PROJECT_ROOT / "dataset" / "landmark_data.csv"
MODEL_OUT     = PROJECT_ROOT / "models" / "gesture" / "gesture_model.h5"
LABELS_OUT    = PROJECT_ROOT / "models" / "gesture" / "labels.txt"
NUM_FEATURES  = 42  # 21 landmarks × 2 (x, y)
MIN_IMAGES    = 5   # minimum images required to include a class

# ── MediaPipe setup ──────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
)

def extract_landmarks(image_path: Path):
    """Return normalised (x,y) landmark list or None if hand not found."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    return [coord for point in lm.landmark for coord in (point.x, point.y)]

# ── Dataset discovery ────────────────────────────────────────────────────────

def discover_classes():
    """Find a dataset root that contains class sub-folders."""
    for base in DATASET_PATHS:
        if base.is_dir():
            subdirs = [d for d in base.iterdir() if d.is_dir()]
            if subdirs:
                print(f"[INFO] Using dataset root: {base}")
                return base, subdirs
    raise FileNotFoundError(
        "Could not find ASL dataset. "
        f"Checked: {[str(p) for p in DATASET_PATHS]}"
    )

# ── Extraction ───────────────────────────────────────────────────────────────

def extract_all(base_dir: Path, class_dirs: list):
    """Walk all class folders, extract landmarks, return X, y, labels."""
    labels = sorted([d.name for d in class_dirs])

    rows_X, rows_y = [], []
    success_total, fail_total = 0, 0

    for label_idx, label in enumerate(labels):
        class_dir = base_dir / label
        images = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

        class_X = []
        for img_path in images:
            feats = extract_landmarks(img_path)
            if feats is not None:
                class_X.append(feats)
            else:
                fail_total += 1

        if len(class_X) < MIN_IMAGES:
            print(f"  [SKIP] '{label}' — only {len(class_X)} detections (< {MIN_IMAGES})")
            continue

        for feat in class_X:
            rows_X.append(feat)
            rows_y.append(label_idx)
        success_total += len(class_X)
        print(f"  [OK]   '{label}' — {len(class_X)}/{len(images)} images extracted")

    print(f"\n[INFO] Total extracted: {success_total} | Failed detections: {fail_total}")
    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.int32), labels

# ── Save CSV ─────────────────────────────────────────────────────────────────

def save_csv(X, y):
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        for features, label in zip(X, y):
            writer.writerow([label] + list(features))
    print(f"[INFO] Saved landmark CSV → {CSV_OUT}")

# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(X, y, labels):
    num_classes = len(labels)
    print(f"\n[INFO] Training on {len(X)} samples, {num_classes} classes …")

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Re-index labels to be contiguous (some may have been skipped)
    unique_classes = sorted(set(y.tolist()))
    remap = {old: new for new, old in enumerate(unique_classes)}
    y = np.array([remap[v] for v in y], dtype=np.int32)
    labels = [labels[c] for c in unique_classes]

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(num_classes=len(labels))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_OUT), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\n[DONE] Best val accuracy: {val_acc:.4f}")
    print(f"[DONE] Model saved → {MODEL_OUT}")

    # Save labels
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_OUT, "w") as f:
        f.write("\n".join(labels))
    print(f"[DONE] Labels saved → {LABELS_OUT}")

    return labels

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Sign Language Recognition — Data Extraction & Training")
    print("=" * 60)

    base_dir, class_dirs = discover_classes()

    print(f"\n[STEP 1] Extracting landmarks from {len(class_dirs)} classes …")
    X, y, labels = extract_all(base_dir, class_dirs)

    if len(X) == 0:
        print("[ERROR] No landmarks extracted. Check dataset path and MediaPipe install.")
        sys.exit(1)

    save_csv(X, y)

    print("\n[STEP 2] Training model …")
    labels = train(X, y, labels)

    print("\n✅ Pipeline complete!")
    print(f"   Classes: {labels}")

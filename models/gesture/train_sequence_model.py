import os
import sys
import numpy as np
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtime_env import configure_runtime_environment
configure_runtime_environment()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Parameters
SEQUENCE_LENGTH = 30
FEATURES = 225  # Pose (33*3) + Left Hand (21*3) + Right Hand (21*3)
MODEL_PATH = str(PROJECT_ROOT / "models" / "gesture" / "lstm_gesture_model.keras")
LABELS_PATH = str(PROJECT_ROOT / "dataset" / "dataset" / "wlasl_features" / "labels.txt")
DATA_PATH = str(PROJECT_ROOT / "dataset" / "dataset" / "wlasl_features" / "sequences")

def build_lstm_model(sequence_length=SEQUENCE_LENGTH, features=FEATURES, num_classes=100):
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=(sequence_length, features)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(256, return_sequences=True, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(128, return_sequences=False, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
        
    label_map = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)
    
    X, y = [], []
    
    for label in labels:
        class_dir = os.path.join(DATA_PATH, label)
        if not os.path.exists(class_dir):
            continue
            
        for npy_file in os.listdir(class_dir):
            if npy_file.endswith('.npy'):
                res = np.load(os.path.join(class_dir, npy_file))
                # WLASL features are expected to be (30, 225)
                if res.shape == (SEQUENCE_LENGTH, FEATURES):
                    X.append(res)
                    y.append(label_map[label])
                else:
                    # Try to pad or truncate if needed, though they should be (30, 225)
                    if res.shape[1] == FEATURES:
                        if res.shape[0] < SEQUENCE_LENGTH:
                            # Pad
                            pad_width = SEQUENCE_LENGTH - res.shape[0]
                            res = np.pad(res, ((0, pad_width), (0, 0)), mode='constant')
                            X.append(res)
                            y.append(label_map[label])
                        elif res.shape[0] > SEQUENCE_LENGTH:
                            # Truncate
                            X.append(res[:SEQUENCE_LENGTH, :])
                            y.append(label_map[label])
                    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Loaded {len(X)} sequences of shape {X.shape[1:]} for {num_classes} classes.")
    return X, y, labels

def train_model():
    print("Loading data...")
    X, y, labels = load_data()
    
    if len(X) == 0:
        print("No valid sequences found. Exiting.")
        return
        
    # Shuffle data
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Split data (85% train, 15% validation)
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Building LSTM model for {len(labels)} classes...")
    model = build_lstm_model(num_classes=len(labels))
    model.summary()
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Save the labels file to models/gesture for prediction
    target_labels_path = os.path.join(os.path.dirname(MODEL_PATH), "labels.txt")
    with open(target_labels_path, 'w') as f:
        f.write("\n".join(labels))
        
    print(f"Training complete. Best model saved to {MODEL_PATH}")
    print(f"Labels saved to {target_labels_path}")

if __name__ == "__main__":
    train_model()

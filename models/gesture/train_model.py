import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

import tensorflow as tf

DATA_PATH = "dataset/gesture_data.csv"
MODEL_OUT = "models/gesture/gesture_model.h5"


def load_dataset(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} was not found. Collect gesture samples first with `python utils/data_collection.py`."
        )
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    X = data[:, 1:].astype(np.float32)
    y = data[:, 0].astype(np.int32)
    return X, y


def split_dataset(X, y, test_ratio=0.2, seed=42):
    if len(X) < 2:
        return X, X, y, y

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split_at = max(1, int(len(indices) * (1 - test_ratio)))
    train_idx = indices[:split_at]
    test_idx = indices[split_at:] if split_at < len(indices) else indices[:1]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


X, y = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=50, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

model.save(MODEL_OUT)

print(f"Gesture model trained and saved to {MODEL_OUT}")
print(f"Evaluation accuracy: {accuracy:.4f}")

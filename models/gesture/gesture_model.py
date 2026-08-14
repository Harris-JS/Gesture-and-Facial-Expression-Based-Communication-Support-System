import os
import numpy as np
from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

try:
    import tensorflow as tf
    import tf_keras
except Exception as e:
    tf = None
    tf_keras = None


class GestureModel:
    def __init__(self, model_path="models/gesture/lstm_gesture_model.keras", labels_path="models/gesture/labels.txt"):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.sequence_length = 30
        self.features_dim = 225
        
        self.labels = []
        self._load_labels()
        self.load_model(model_path)

    def _load_labels(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
        if not self.labels:
            # Fallback
            self.labels = [str(i) for i in range(100)]

    def load_model(self, path=None):
        target_path = path or self.model_path
        if tf is None or not os.path.exists(target_path):
            self.model = None
            return None

        try:
            # Use tf_keras loader (matches training environment)
            self.model = tf_keras.models.load_model(target_path)
            self._warmup()
        except Exception as e:
            print(f"[GestureModel] Failed to load model: {e}")
            self.model = None
        return self.model

    def _warmup(self):
        if self.model is None:
            return
        dummy = np.zeros(
            (1, self.sequence_length, self.features_dim), dtype=np.float32
        )
        self.model.predict(dummy, verbose=0)

    def predict_with_confidence(self, feature_sequence):
        if self.model is None or feature_sequence is None:
            return None, 0.0

        if len(feature_sequence) < self.sequence_length:
            return None, 0.0 # Wait until buffer is full

        # feature_sequence should be a list or array of shape (30, 42)
        sequence = np.array(feature_sequence, dtype=np.float32)
        # Reshape for LSTM (batch_size, sequence_length, features)
        sequence = sequence.reshape(1, self.sequence_length, self.features_dim)
        
        prediction = self.model.predict(sequence, verbose=0)
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[0][class_id])

        if class_id < len(self.labels):
            return self.labels[class_id], confidence
        return str(class_id), confidence

    def predict(self, feature_sequence):
        label, _ = self.predict_with_confidence(feature_sequence)
        return label

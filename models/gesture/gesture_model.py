import os
import numpy as np
from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

try:
    import tensorflow as tf
except Exception:
    tf = None


class GestureModel:
    def __init__(self, model_path="models/gesture/gesture_model.h5"):
        self.model_path = model_path
        self.model = None
        self.labels = [
            "Hello",
            "Yes",
            "No",
            "Thank You",
            "Help",
        ]
        self.load_model(model_path)

    def load_model(self, path=None):
        target_path = path or self.model_path
        if tf is None or not os.path.exists(target_path):
            self.model = None
            return None

        self.model = tf.keras.models.load_model(target_path)
        return self.model

    def predict_with_confidence(self, features):
        if self.model is None or features is None:
            return None, 0.0

        features = np.array(features, dtype=np.float32).reshape(1, -1)
        prediction = self.model.predict(features, verbose=0)
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[0][class_id])

        if class_id < len(self.labels):
            return self.labels[class_id], confidence
        return str(class_id), confidence

    def predict(self, features):
        label, _ = self.predict_with_confidence(features)
        return label

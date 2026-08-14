import os
import numpy as np

from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

class EmotionModel:
    def __init__(self, **kwargs):
        self.labels = DEFAULT_LABELS.copy()
        self.backend = "deepface" if DeepFace is not None else None

    def predict_scores(self, face_image):
        if face_image is None or self.backend != "deepface":
            return None

        try:
            # We already have an RGB image from cvtColor
            if len(face_image.shape) == 3:
                # Deepface usually expects BGR array, but we pass what we have
                bgr_image = face_image[:, :, ::-1]
            else:
                bgr_image = face_image

            result = DeepFace.analyze(
                img_path=bgr_image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace can return a list if multiple faces are found (enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
                
            emotions = result.get('emotion', {})
            
            # Convert to our labels array order
            scores = np.zeros(len(self.labels), dtype=np.float32)
            for i, label in enumerate(self.labels):
                scores[i] = emotions.get(label, 0.0) / 100.0 # DeepFace returns percentages 0-100
                
            return scores
        except Exception as e:
            print(f"[EmotionModel] DeepFace analyze error: {e}")
            return None

    def predict_with_confidence(self, face_image):
        scores = self.predict_scores(face_image)
        if scores is None:
            return None, 0.0

        index = int(np.argmax(scores))
        confidence = float(scores[index])

        if index < len(self.labels):
            return self.labels[index], confidence
        return str(index), confidence

    def predict(self, face_image):
        label, _ = self.predict_with_confidence(face_image)
        return label

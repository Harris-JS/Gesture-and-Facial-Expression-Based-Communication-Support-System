import os

import numpy as np

from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except Exception:
    torch = None
    Image = None
    AutoImageProcessor = None
    AutoModelForImageClassification = None


DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEFAULT_HF_MODEL_ID = "trpakov/vit-face-expression"


class EmotionModel:
    def __init__(
        self,
        model_path="models/emotion/emotion_model.h5",
        labels_path=None,
        hf_model_id=DEFAULT_HF_MODEL_ID,
    ):
        self.model_path = model_path
        self.hf_model_id = hf_model_id
        self.model = None
        self.labels = DEFAULT_LABELS.copy()
        self.backend = None
        self.processor = None

        if labels_path and os.path.exists(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as handle:
                    loaded = [line.strip() for line in handle.readlines() if line.strip()]
                if loaded:
                    self.labels = loaded
            except Exception:
                pass

        self._load_best_available_model()

    def _load_best_available_model(self):
        if AutoImageProcessor is not None and AutoModelForImageClassification is not None and torch is not None:
            for local_only in (True, False):
                try:
                    self.processor = AutoImageProcessor.from_pretrained(
                        self.hf_model_id,
                        local_files_only=local_only,
                    )
                    self.model = AutoModelForImageClassification.from_pretrained(
                        self.hf_model_id,
                        local_files_only=local_only,
                    )
                    self.model.eval()
                    id2label = getattr(self.model.config, "id2label", {}) or {}
                    loaded_labels = [id2label[key] for key in sorted(id2label.keys())]
                    if loaded_labels:
                        self.labels = [label.lower() for label in loaded_labels]
                    self.backend = "huggingface"
                    return
                except Exception:
                    self.processor = None
                    self.model = None

        if tf is not None and os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.backend = "keras"
            except Exception:
                self.model = None
                self.backend = None

    @staticmethod
    def preprocess_face(face_image, target_size=(48, 48), as_rgb=False):
        if face_image is None:
            return None

        try:
            import cv2
        except Exception:
            return None

        if getattr(face_image, "size", 0) == 0:
            return None

        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        resized = cv2.resize(gray, target_size)

        if as_rgb:
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        resized = resized.astype(np.float32) / 255.0
        return np.expand_dims(resized, axis=-1)

    def predict_scores(self, face_image):
        if face_image is None or self.model is None:
            return None

        if self.backend == "huggingface":
            if Image is None or self.processor is None or torch is None:
                return None

            if len(face_image.shape) == 2:
                pil_image = Image.fromarray(face_image)
            else:
                pil_image = Image.fromarray(face_image[:, :, ::-1])

            inputs = self.processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits

            probabilities = torch.softmax(logits, dim=-1)[0]
            return probabilities.detach().cpu().numpy().astype(np.float32)

        processed = self.preprocess_face(face_image)
        if processed is None:
            return None

        batch = np.expand_dims(processed, axis=0)
        predictions = self.model.predict(batch, verbose=0)
        return predictions[0].astype(np.float32)

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

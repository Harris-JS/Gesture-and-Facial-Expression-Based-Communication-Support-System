from dataclasses import dataclass, field
from collections import deque
from typing import Any, List, Optional

import numpy as np


def fuse_predictions(gesture: Optional[str], emotion: Optional[str]) -> Optional[str]:
    parts = []
    if gesture:
        parts.append(str(gesture))
    if emotion:
        parts.append(str(emotion))
    return " + ".join(parts) if parts else None


def build_output_message(
    gesture: Optional[str],
    emotion: Optional[str],
    hand_detected: bool = False,
    face_detected: bool = False,
) -> Optional[str]:
    fused = fuse_predictions(gesture, emotion)
    if fused:
        return fused

    if hand_detected and face_detected:
        return "Hand and face detected"
    if hand_detected:
        return "Hand detected"
    if face_detected:
        return "Face detected"
    return None


@dataclass
class FramePrediction:
    frame: Any
    timestamp: float
    gesture: Optional[str] = None
    emotion: Optional[str] = None
    gesture_confidence: float = 0.0
    emotion_confidence: float = 0.0
    emotion_candidates: List[str] = field(default_factory=list)
    output: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def apply_prediction_overlay(frame: Any, prediction: FramePrediction) -> Any:
    try:
        import cv2
    except Exception:
        return frame

    if prediction.gesture:
        cv2.putText(
            frame,
            f"Gesture: {prediction.gesture} ({prediction.gesture_confidence:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
    if prediction.emotion:
        cv2.putText(
            frame,
            f"Emotion: {prediction.emotion} ({prediction.emotion_confidence:.2f})",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    if prediction.output:
        cv2.putText(
            frame,
            f"Output: {prediction.output}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
    return frame


class AssistiveCommunicationRuntime:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.video = None
        self.preprocess = None
        self.feature_extractor = None
        self.gesture_model = None
        self.emotion_model = None
        self.warnings: List[str] = []
        self._label_state = {
            "gesture": {"candidate": None, "streak": 0, "stable": None},
            "emotion": {"candidate": None, "streak": 0, "stable": None},
        }
        self._emotion_scores = deque(maxlen=5)
        self._initialize_components()

    @staticmethod
    def _extract_aligned_face(frame: Any, face_landmarks: Any, face_bbox: Any) -> Any:
        if face_bbox is None:
            return None

        x, y, bw, bh = face_bbox
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame.shape[1], x + bw)
        y1 = min(frame.shape[0], y + bh)
        if x1 <= x0 or y1 <= y0:
            return None

        if face_landmarks is None:
            return frame[y0:y1, x0:x1]

        try:
            import cv2
        except Exception:
            return frame[y0:y1, x0:x1]

        # MediaPipe face mesh eye corner landmarks.
        left_eye_indices = [33, 133]
        right_eye_indices = [362, 263]

        def _center(indices):
            xs = [face_landmarks.landmark[index].x * frame.shape[1] for index in indices]
            ys = [face_landmarks.landmark[index].y * frame.shape[0] for index in indices]
            return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

        left_eye = _center(left_eye_indices)
        right_eye = _center(right_eye_indices)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        if dx == 0 and dy == 0:
            return frame[y0:y1, x0:x1]

        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned = cv2.warpAffine(
            frame,
            matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        corners = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float32,
        )
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        transformed = np.hstack([corners, ones]) @ matrix.T
        tx0 = max(0, int(np.min(transformed[:, 0])))
        ty0 = max(0, int(np.min(transformed[:, 1])))
        tx1 = min(aligned.shape[1], int(np.max(transformed[:, 0])))
        ty1 = min(aligned.shape[0], int(np.max(transformed[:, 1])))
        if tx1 <= tx0 or ty1 <= ty0:
            return frame[y0:y1, x0:x1]

        return aligned[ty0:ty1, tx0:tx1]

    def _format_top_candidates(self, scores, labels, limit=3):
        if scores is None:
            return []
        ranked = np.argsort(scores)[::-1][:limit]
        return [f"{labels[index]}:{float(scores[index]):.2f}" for index in ranked if index < len(labels)]

    def _stabilize_label(
        self,
        key: str,
        detected: bool,
        candidate: Optional[str],
        confidence: float,
        threshold: float,
        required_hits: int,
    ) -> Optional[str]:
        state = self._label_state[key]

        if not detected:
            state["candidate"] = None
            state["streak"] = 0
            state["stable"] = None
            return None

        if candidate is None or confidence < threshold:
            state["candidate"] = None
            state["streak"] = 0
            return None

        if state["stable"] == candidate:
            state["candidate"] = candidate
            state["streak"] = required_hits
            return candidate

        if state["candidate"] == candidate:
            state["streak"] += 1
        else:
            state["candidate"] = candidate
            state["streak"] = 1

        if state["streak"] >= required_hits:
            state["stable"] = candidate
            return candidate

        return None

    def _initialize_components(self) -> None:
        try:
            from utils.input_module import VideoCaptureModule

            self.video = VideoCaptureModule(camera_index=self.camera_index)
        except Exception as exc:
            self.warnings.append(f"Camera unavailable: {exc}")
            return

        try:
            from utils.preprocessing import PreprocessingModule
            from utils.feature_extraction import FeatureExtractionModule

            self.preprocess = PreprocessingModule()
            self.feature_extractor = FeatureExtractionModule()
        except Exception as exc:
            self.warnings.append(f"Preprocessing unavailable: {exc}")

        try:
            from models.gesture.gesture_model import GestureModel

            self.gesture_model = GestureModel()
            if self.gesture_model.model is None:
                self.warnings.append(
                    "Gesture model file not loaded. Train one with models/gesture/train_model.py."
                )
        except Exception as exc:
            self.warnings.append(f"Gesture model unavailable: {exc}")

        try:
            from models.emotion.emotion_model import EmotionModel

            self.emotion_model = EmotionModel(labels_path="models/emotion/labels.txt")
            if self.emotion_model.model is None:
                self.warnings.append(
                    "Emotion model file not loaded. Train one with models/emotion/train_emotion_model.py."
                )
            elif getattr(self.emotion_model, "backend", None) == "huggingface":
                self.warnings.append("Emotion model backend: Hugging Face ViT face-expression model.")
        except Exception as exc:
            self.warnings.append(f"Emotion model unavailable: {exc}")

    @property
    def ready(self) -> bool:
        return self.video is not None

    def _smooth_emotion(self, detected: bool, scores):
        if not detected or scores is None:
            self._emotion_scores.clear()
            return None, 0.0

        self._emotion_scores.append(np.array(scores, dtype=np.float32))
        averaged = np.mean(np.stack(self._emotion_scores, axis=0), axis=0)
        top_index = int(np.argmax(averaged))
        top_confidence = float(averaged[top_index])
        sorted_scores = np.sort(averaged)
        second_confidence = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
        margin = top_confidence - second_confidence

        if top_index >= len(self.emotion_model.labels):
            return None, 0.0

        label = self.emotion_model.labels[top_index]
        if label == "neutral":
            if top_confidence < 0.75 or margin < 0.18:
                return None, top_confidence
        else:
            if top_confidence < 0.50 or margin < 0.08:
                return None, top_confidence

        return label, top_confidence

    def read_prediction(self) -> Optional[FramePrediction]:
        if self.video is None:
            raise RuntimeError("Video capture is not initialized.")

        result = self.video.get_frame()
        if result is None:
            return None

        frame, timestamp = result
        raw_frame = frame.copy()
        hand_landmarks = None
        face_landmarks = None
        face_bbox = None
        face_area_ratio = 0.0

        if self.preprocess is not None:
            frame, hand_landmarks, face_landmarks, face_bbox = self.preprocess.process(frame)

        hand_features = None
        face_image = None
        if self.feature_extractor is not None:
            hand_features = self.feature_extractor.extract_hand_features(hand_landmarks, frame.shape)
        if face_bbox is not None:
            x, y, bw, bh = face_bbox
            face_area_ratio = (max(0, bw) * max(0, bh)) / float(raw_frame.shape[0] * raw_frame.shape[1])
            face_image = self._extract_aligned_face(raw_frame, face_landmarks, face_bbox)

        raw_gesture = None
        raw_emotion = None
        gesture_confidence = 0.0
        emotion_confidence = 0.0
        emotion_scores = None
        if self.gesture_model is not None:
            raw_gesture, gesture_confidence = self.gesture_model.predict_with_confidence(hand_features)
        if self.emotion_model is not None and face_area_ratio >= 0.05:
            emotion_scores = self.emotion_model.predict_scores(face_image)
            raw_emotion, emotion_confidence = self.emotion_model.predict_with_confidence(face_image)

        hand_detected = hand_landmarks is not None
        face_detected = face_landmarks is not None
        gesture = self._stabilize_label(
            key="gesture",
            detected=hand_detected,
            candidate=raw_gesture,
            confidence=gesture_confidence,
            threshold=0.60,
            required_hits=3,
        )
        emotion, emotion_confidence = self._smooth_emotion(face_detected, emotion_scores)

        return FramePrediction(
            frame=frame,
            timestamp=timestamp,
            gesture=gesture,
            emotion=emotion,
            gesture_confidence=gesture_confidence,
            emotion_confidence=emotion_confidence,
            emotion_candidates=self._format_top_candidates(
                emotion_scores,
                self.emotion_model.labels if self.emotion_model is not None else [],
            ),
            output=build_output_message(
                gesture=gesture,
                emotion=emotion,
                hand_detected=hand_detected,
                face_detected=face_detected,
            ),
            warnings=list(self.warnings),
        )

    def release(self) -> None:
        if self.video is not None:
            self.video.release()

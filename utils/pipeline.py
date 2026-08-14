from dataclasses import dataclass, field
from collections import deque
from typing import Any, List, Optional
import os
import threading
import time

import numpy as np


# ---------------------------------------------------------------------------
# Emotion-Aware Semantic Fusion
# ---------------------------------------------------------------------------

# Maps emotion -> (sentence_ending, emoji, prefix_exclamation)
_EMOTION_RULES = {
    "happy":    ("! 😊",   ""),
    "sad":      ("... 😔",  ""),
    "surprise": ("? 😲",   "Oh! "),
    "angry":    (". 😠",    ""),
    "fear":     ("... 😨",  ""),
    "disgust":  (". 😒",    ""),
    "neutral":  (".",       ""),
}


def apply_emotion_semantics(sentence: str, emotion: Optional[str]) -> str:
    """Take a raw accumulated sentence and emotion, return a naturally
    punctuated, emoji-annotated interpretation string."""
    if not sentence:
        return sentence

    # Strip trailing whitespace / existing punctuation for clean insertion
    cleaned = sentence.strip().rstrip(".!?,")

    if not emotion:
        return cleaned + "."

    key = emotion.lower()
    ending, prefix = _EMOTION_RULES.get(key, (".", ""))

    # Capitalise first word
    words = cleaned.split()
    if words:
        words[0] = words[0].capitalize()
    cleaned = " ".join(words)

    return f"{prefix}{cleaned}{ending}"


# Legacy shim – kept so existing unit-tests that import fuse_predictions still pass
def fuse_predictions(gesture: Optional[str], emotion: Optional[str]) -> Optional[str]:
    parts = []
    if gesture:
        parts.append(str(gesture).capitalize())
    if emotion:
        parts.append(f"({str(emotion).capitalize()})")
    return " ".join(parts) if parts else None


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


# ---------------------------------------------------------------------------
# Sentence Accumulator
# ---------------------------------------------------------------------------

class SentenceAccumulator:
    """Accumulates stabilised gesture labels into a running sentence.

    Rules
    -----
    * A new word is appended only when the gesture changes (no duplicate
      repetition of the same label).
    * If no new word arrives within *timeout_sec* seconds the sentence is
      considered complete and the buffer is cleared so the next sign starts
      a fresh sentence.
    * Maximum sentence length is capped at *max_words* to prevent unbounded
      growth during demos.
    """

    def __init__(self, timeout_sec: float = 4.0, max_words: int = 12):
        self._words: List[str] = []
        self._last_word: Optional[str] = None
        self._last_update: float = 0.0
        self.timeout_sec = timeout_sec
        self.max_words = max_words

    def push(self, word: Optional[str]) -> None:
        """Feed the latest stabilised gesture label."""
        now = time.monotonic()

        # Auto-clear if idle for too long
        if self._words and (now - self._last_update) > self.timeout_sec:
            self._words = []
            self._last_word = None

        if not word:
            return

        # Append only on change
        if word != self._last_word:
            if len(self._words) < self.max_words:
                self._words.append(word.lower())
            self._last_word = word
            self._last_update = now

    def sentence(self) -> str:
        return " ".join(self._words)

    def clear(self) -> None:
        self._words = []
        self._last_word = None


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FramePrediction:
    frame: Any
    timestamp: float
    gesture: Optional[str] = None
    emotion: Optional[str] = None
    gesture_confidence: float = 0.0
    emotion_confidence: float = 0.0
    emotion_candidates: List[str] = field(default_factory=list)
    # Raw accumulated sign words (space-separated, e.g. "hello how are you")
    accumulated_sentence: str = ""
    # Emotion-aware interpreted sentence (e.g. "Hello! How are you? 😊")
    interpreted_sentence: str = ""
    # Translated version of interpreted_sentence
    translated_output: Optional[str] = None
    # Legacy field kept for backwards-compat
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
            f"Sign: {prediction.gesture} ({prediction.gesture_confidence:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )
    if prediction.emotion:
        cv2.putText(
            frame,
            f"Emotion: {prediction.emotion} ({prediction.emotion_confidence:.2f})",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 100),
            2,
        )
    if prediction.interpreted_sentence:
        display = (
            prediction.translated_output
            if prediction.translated_output
            else prediction.interpreted_sentence
        )
        # Truncate for overlay
        if len(display) > 55:
            display = display[:52] + "..."
        cv2.putText(
            frame,
            display,
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
    return frame


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class AssistiveCommunicationRuntime:
    def __init__(
        self,
        camera_index: int = 0,
        *,
        enable_emotion: bool = True,
        fast_mode: bool = False,
    ):
        self.camera_index = camera_index
        self.enable_emotion = enable_emotion
        self.fast_mode = fast_mode
        self.video = None
        self.preprocess = None
        self.feature_extractor = None
        self.gesture_model = None
        self.emotion_model = None
        self.translator = None
        self.speech = None

        # Configuration
        self.target_language = "en"
        self.tts_enabled = False
        self.last_spoken_output: Optional[str] = None

        self.warnings: List[str] = []
        self._label_state = {
            "gesture": {"candidate": None, "streak": 0, "stable": None},
            "emotion": {"candidate": None, "streak": 0, "stable": None},
        }
        self._emotion_scores = deque(maxlen=5)
        self._gesture_sequence_length = 30
        self._gesture_buffer = deque(maxlen=30)
        self._gesture_threshold = 0.35 if fast_mode else 0.60
        self._gesture_hits = 2 if fast_mode else 3
        self._gesture_load_started = False
        self._gesture_load_lock = threading.Lock()
        self._accumulator = SentenceAccumulator(timeout_sec=4.0)
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
        return frame[y0:y1, x0:x1]

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
            max_idx = 1 if self.fast_mode else 4
            backend = "DSHOW" if self.fast_mode and os.name == "nt" else None
            self.video = VideoCaptureModule(
                camera_index=self.camera_index,
                max_camera_index=max_idx,
                backend=backend,
            )
        except Exception as exc:
            self.warnings.append(f"Camera unavailable: {exc}")
            return

        if self.fast_mode:
            # Web: open camera first; load MediaPipe/model on first processed frame.
            return

        self._init_ml_components()

    def _init_ml_components(self) -> None:
        try:
            from utils.preprocessing import PreprocessingModule
            from utils.feature_extraction import FeatureExtractionModule
            self.preprocess = PreprocessingModule(fast_mode=self.fast_mode)
            self.feature_extractor = FeatureExtractionModule()
        except Exception as exc:
            self.warnings.append(f"Preprocessing unavailable: {exc}")

        if not self.fast_mode:
            self._load_gesture_model()

        if self.enable_emotion:
            try:
                from models.emotion.emotion_model import EmotionModel
                self.emotion_model = EmotionModel(labels_path="models/emotion/labels.txt")
                if getattr(self.emotion_model, "backend", None) is None:
                    self.warnings.append("Emotion model backend could not be loaded.")
            except Exception as exc:
                self.warnings.append(f"Emotion model unavailable: {exc}")

        try:
            from utils.translation import TranslationModule
            self.translator = TranslationModule()
        except Exception as exc:
            self.warnings.append(f"Translation module unavailable: {exc}")

        if not self.fast_mode:
            try:
                from utils.speech import SpeechModule
                self.speech = SpeechModule()
            except Exception as exc:
                self.warnings.append(f"Speech module unavailable: {exc}")

    def _load_gesture_model(self) -> None:
        try:
            from models.gesture.gesture_model import GestureModel

            self.gesture_model = GestureModel()
            if self.gesture_model.model is None:
                self.warnings.append(
                    "Gesture model file not loaded. Train one with models/gesture/train_model.py."
                )
        except Exception as exc:
            self.warnings.append(f"Gesture model unavailable: {exc}")

    def ensure_gesture_model_async(self) -> None:
        if self.gesture_model is not None:
            return
        with self._gesture_load_lock:
            if self._gesture_load_started:
                return
            self._gesture_load_started = True

        def _load():
            self._load_gesture_model()

        threading.Thread(target=_load, daemon=True).start()

    def ensure_ml_ready(self) -> None:
        """Lazy-load MediaPipe and feature extraction (web fast startup)."""
        if self.preprocess is not None:
            return
        self._init_ml_components()
        if self.fast_mode:
            self.ensure_gesture_model_async()

    @property
    def ready(self) -> bool:
        return self.video is not None

    def _smooth_emotion(self, detected: bool, scores):
        if not detected or scores is None:
            return None, 0.0
        top_index = int(np.argmax(scores))
        top_confidence = float(scores[top_index])
        label = self.emotion_model.labels[top_index]
        return label, top_confidence

    def read_prediction(self) -> Optional[FramePrediction]:
        if self.video is None:
            raise RuntimeError("Video capture is not initialized.")

        result = self.video.get_frame()
        if result is None:
            return None

        frame, timestamp = result
        return self.process_frame(frame, timestamp)

    def process_frame(self, frame: Any, timestamp: float) -> Optional[FramePrediction]:
        self.ensure_ml_ready()
        raw_frame = frame.copy()
        holistic_results = None
        face_landmarks = None
        face_bbox = None
        face_area_ratio = 0.0

        if self.preprocess is None:
            return FramePrediction(frame=frame, timestamp=timestamp)

        frame, holistic_results, face_landmarks, face_bbox = self.preprocess.process(frame)

        hand_features = None
        face_image = None
        if self.feature_extractor is not None and holistic_results is not None:
            hand_features = self.feature_extractor.extract_holistic_features(
                holistic_results.pose_landmarks,
                holistic_results.left_hand_landmarks,
                holistic_results.right_hand_landmarks,
            )
        if face_bbox is not None:
            x, y, bw, bh = face_bbox
            face_area_ratio = (max(0, bw) * max(0, bh)) / float(
                raw_frame.shape[0] * raw_frame.shape[1]
            )
            face_image = self._extract_aligned_face(raw_frame, face_landmarks, face_bbox)

        raw_gesture = None
        raw_emotion = None
        gesture_confidence = 0.0
        emotion_confidence = 0.0
        emotion_scores = None

        # Buffer hand features for LSTM sequence model
        if hand_features is not None:
            self._gesture_buffer.append(hand_features)
        else:
            self._gesture_buffer.append([0.0] * 225)

        hand_detected = holistic_results is not None and (
            holistic_results.left_hand_landmarks is not None
            or holistic_results.right_hand_landmarks is not None
        )

        if (
            self.gesture_model is not None
            and hand_detected
            and len(self._gesture_buffer) == self._gesture_sequence_length
        ):
            raw_gesture, gesture_confidence = self.gesture_model.predict_with_confidence(
                list(self._gesture_buffer)
            )

        if self.emotion_model is not None and face_area_ratio >= 0.01:
            emotion_scores = self.emotion_model.predict_scores(face_image)
            if emotion_scores is not None:
                top_index = int(np.argmax(emotion_scores))
                emotion_confidence = float(emotion_scores[top_index])
                if top_index < len(self.emotion_model.labels):
                    raw_emotion = self.emotion_model.labels[top_index]

        face_detected = face_bbox is not None

        gesture = self._stabilize_label(
            key="gesture",
            detected=hand_detected,
            candidate=raw_gesture,
            confidence=gesture_confidence,
            threshold=self._gesture_threshold,
            required_hits=self._gesture_hits,
        )
        # In fast/web mode, show the live model label while stabilizer warms up.
        if gesture is None and raw_gesture and gesture_confidence >= 0.20:
            gesture = raw_gesture

        emotion, emotion_confidence = self._smooth_emotion(face_detected, emotion_scores)

        # ---- Sentence Accumulation ----------------------------------------
        word = gesture
        if word is None and raw_gesture and gesture_confidence >= 0.25:
            word = raw_gesture
        self._accumulator.push(word)
        accumulated = self._accumulator.sentence()

        # ---- Emotion-Aware Interpretation ------------------------------------
        interpreted = apply_emotion_semantics(accumulated, emotion) if accumulated else ""

        # ---- Translation -----------------------------------------------------
        translated = interpreted
        if (
            interpreted
            and self.target_language != "en"
            and self.translator is not None
        ):
            translated = self.translator.translate(interpreted, self.target_language)

        # ---- TTS (speak only when interpretation changes) --------------------
        if (
            self.tts_enabled
            and self.speech
            and interpreted
            and interpreted != self.last_spoken_output
            and (gesture or emotion)
        ):
            speak_text = translated if translated else interpreted
            self.speech.speak(speak_text)
            self.last_spoken_output = interpreted
        elif not interpreted:
            self.last_spoken_output = None

        # Legacy 'output' kept for backward compat
        legacy_output = build_output_message(
            gesture=gesture,
            emotion=emotion,
            hand_detected=hand_detected,
            face_detected=face_detected,
        )

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
            accumulated_sentence=accumulated,
            interpreted_sentence=interpreted,
            translated_output=translated if self.target_language != "en" else None,
            output=legacy_output,
            warnings=list(self.warnings),
        )

    def release(self) -> None:
        if self.video is not None:
            self.video.release()
        if self.speech is not None:
            self.speech.shutdown()

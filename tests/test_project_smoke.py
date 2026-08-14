import os
import sys
import unittest

# Add scripts directory to sys.path so verify_project can be imported
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import web_app
import verify_project
from utils.pipeline import (
    build_output_message,
    fuse_predictions,
    apply_emotion_semantics,
    SentenceAccumulator,
)


class FakeRuntime:
    def __init__(self, ready, warnings=None):
        self.ready = ready
        self.warnings = warnings or []
        self.target_language = "en"
        self.tts_enabled = False

    def release(self):
        return None


class ProjectSmokeTests(unittest.TestCase):

    # ── legacy fuse_predictions shim ─────────────────────────────────────────
    def test_fuse_predictions_with_both_inputs(self):
        self.assertEqual(fuse_predictions("Hello", "happy"), "Hello (Happy)")

    def test_fuse_predictions_with_missing_values(self):
        self.assertEqual(fuse_predictions("Help", None), "Help")
        self.assertEqual(fuse_predictions(None, "sad"), "(Sad)")
        self.assertIsNone(fuse_predictions(None, None))

    # ── emotion-aware semantic fusion ─────────────────────────────────────────
    def test_emotion_semantics_happy(self):
        result = apply_emotion_semantics("hello how are you", "happy")
        self.assertIn("😊", result)
        self.assertIn("!", result)

    def test_emotion_semantics_sad(self):
        result = apply_emotion_semantics("hello how are you", "sad")
        self.assertIn("😔", result)

    def test_emotion_semantics_surprise(self):
        result = apply_emotion_semantics("hello how are you", "surprise")
        self.assertIn("Oh!", result)
        self.assertIn("😲", result)

    def test_emotion_semantics_no_emotion(self):
        result = apply_emotion_semantics("hello", None)
        # Should end with period; capitalisation follows sentence rules
        self.assertTrue(result.endswith("."))
        self.assertIn("hello", result.lower())

    def test_emotion_semantics_empty_sentence(self):
        self.assertEqual(apply_emotion_semantics("", "happy"), "")

    # ── sentence accumulator ─────────────────────────────────────────────────
    def test_accumulator_appends_new_words(self):
        acc = SentenceAccumulator()
        acc.push("hello")
        acc.push("how")
        acc.push("are")
        self.assertEqual(acc.sentence(), "hello how are")

    def test_accumulator_ignores_duplicate_consecutive_word(self):
        acc = SentenceAccumulator()
        acc.push("hello")
        acc.push("hello")  # same word repeated — should not double-append
        acc.push("you")
        self.assertEqual(acc.sentence(), "hello you")

    def test_accumulator_clear(self):
        acc = SentenceAccumulator()
        acc.push("hello")
        acc.clear()
        self.assertEqual(acc.sentence(), "")

    # ── detection fallback ────────────────────────────────────────────────────
    def test_build_output_message_falls_back_to_detection_status(self):
        self.assertEqual(build_output_message(None, None, hand_detected=True), "Hand detected")
        self.assertEqual(build_output_message(None, None, face_detected=True), "Face detected")
        self.assertEqual(
            build_output_message(None, None, hand_detected=True, face_detected=True),
            "Hand and face detected",
        )

    # ── file existence ────────────────────────────────────────────────────────
    def test_web_interface_files_exist(self):
        self.assertTrue(os.path.exists("web_app.py"))
        self.assertTrue(os.path.exists("templates/index.html"))
        self.assertTrue(os.path.exists("static/styles.css"))

    # ── verify_project integration ────────────────────────────────────────────
    def test_verify_project_python_range_message(self):
        major = sys.version_info.major
        minor = sys.version_info.minor
        supported = (major, minor) >= (3, 10) and (major, minor) <= (3, 12)
        self.assertIsInstance(supported, bool)

    def test_verify_project_runtime_probe_warns_when_camera_missing(self):
        ok, messages = verify_project.check_runtime_probe(
            runtime_class=lambda: FakeRuntime(False, ["Camera unavailable: test camera is offline"])
        )
        self.assertTrue(ok)
        self.assertIn("camera access is unavailable", messages[0].lower())
        self.assertIn("Camera unavailable", messages[1])

    # ── Flask routes ──────────────────────────────────────────────────────────
    def test_web_status_route_returns_cached_runtime_state(self):
        original_runtime = web_app._runtime
        web_app._runtime = FakeRuntime(False, ["Camera unavailable: mocked"])
        try:
            client = web_app.app.test_client()
            response = client.get("/status")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.get_json(),
                {
                    "ready": False,
                    "warnings": ["Camera unavailable: mocked"],
                    "target_language": "en",
                    "tts_enabled": False,
                },
            )
        finally:
            web_app._runtime = original_runtime

    def test_prediction_route_returns_latest_prediction_payload(self):
        original_runtime = web_app._runtime
        original_prediction = dict(web_app._latest_prediction)
        web_app._runtime = FakeRuntime(True, [])
        web_app._latest_prediction = {
            "gesture": "Hello",
            "emotion": "happy",
            "gesture_confidence": 0.91,
            "emotion_confidence": 0.67,
            "emotion_candidates": ["happy:0.67", "neutral:0.20", "surprise:0.08"],
            "accumulated_sentence": "hello",
            "interpreted_sentence": "Hello! 😊",
            "translated_output": None,
            "output": "Hello (Happy)",
            "timestamp": 123.0,
        }
        try:
            client = web_app.app.test_client()
            response = client.get("/prediction")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["gesture"], "Hello")
            self.assertEqual(payload["interpreted_sentence"], "Hello! 😊")
            self.assertEqual(payload["accumulated_sentence"], "hello")
            self.assertTrue(payload["ready"])
        finally:
            web_app._runtime = original_runtime
            web_app._latest_prediction = original_prediction

    def test_clear_sentence_route(self):
        original_runtime = web_app._runtime
        web_app._runtime = FakeRuntime(True, [])
        web_app._latest_prediction["accumulated_sentence"] = "hello how are you"
        web_app._latest_prediction["interpreted_sentence"] = "Hello! How are you? 😊"
        try:
            client = web_app.app.test_client()
            response = client.post("/clear_sentence")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "cleared")
            self.assertEqual(web_app._latest_prediction["accumulated_sentence"], "")
            self.assertEqual(web_app._latest_prediction["interpreted_sentence"], "")
        finally:
            web_app._runtime = original_runtime


if __name__ == "__main__":
    unittest.main()

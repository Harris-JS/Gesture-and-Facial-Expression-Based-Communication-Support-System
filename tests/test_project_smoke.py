import os
import sys
import unittest

import web_app
import verify_project
from utils.pipeline import build_output_message, fuse_predictions


class FakeRuntime:
    def __init__(self, ready, warnings=None):
        self.ready = ready
        self.warnings = warnings or []

    def release(self):
        return None


class ProjectSmokeTests(unittest.TestCase):
    def test_fuse_predictions_with_both_inputs(self):
        self.assertEqual(fuse_predictions("Hello", "happy"), "Hello + happy")

    def test_fuse_predictions_with_missing_values(self):
        self.assertEqual(fuse_predictions("Help", None), "Help")
        self.assertEqual(fuse_predictions(None, "sad"), "sad")
        self.assertIsNone(fuse_predictions(None, None))

    def test_build_output_message_falls_back_to_detection_status(self):
        self.assertEqual(build_output_message(None, None, hand_detected=True), "Hand detected")
        self.assertEqual(build_output_message(None, None, face_detected=True), "Face detected")
        self.assertEqual(
            build_output_message(None, None, hand_detected=True, face_detected=True),
            "Hand and face detected",
        )

    def test_web_interface_files_exist(self):
        self.assertTrue(os.path.exists("web_app.py"))
        self.assertTrue(os.path.exists("templates/index.html"))
        self.assertTrue(os.path.exists("static/styles.css"))

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
            "output": "Hello + happy",
            "timestamp": 123.0,
        }
        try:
            client = web_app.app.test_client()
            response = client.get("/prediction")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.get_json(),
                {
                    "gesture": "Hello",
                    "emotion": "happy",
                    "gesture_confidence": 0.91,
                    "emotion_confidence": 0.67,
                    "emotion_candidates": ["happy:0.67", "neutral:0.20", "surprise:0.08"],
                    "output": "Hello + happy",
                    "timestamp": 123.0,
                    "ready": True,
                    "warnings": [],
                },
            )
        finally:
            web_app._runtime = original_runtime
            web_app._latest_prediction = original_prediction


if __name__ == "__main__":
    unittest.main()

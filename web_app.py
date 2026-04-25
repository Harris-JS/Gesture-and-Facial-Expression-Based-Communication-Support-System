import os
import threading

from flask import Flask, Response, jsonify, render_template

from utils.pipeline import AssistiveCommunicationRuntime, apply_prediction_overlay

app = Flask(__name__)
_runtime = None
_runtime_lock = threading.Lock()
_latest_prediction = {
    "gesture": None,
    "emotion": None,
    "gesture_confidence": 0.0,
    "emotion_confidence": 0.0,
    "emotion_candidates": [],
    "output": None,
    "timestamp": None,
}
_prediction_lock = threading.Lock()


def get_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = AssistiveCommunicationRuntime()
        return _runtime


def update_latest_prediction(prediction):
    global _latest_prediction
    with _prediction_lock:
        _latest_prediction = {
            "gesture": prediction.gesture,
            "emotion": prediction.emotion,
            "gesture_confidence": round(float(prediction.gesture_confidence), 3),
            "emotion_confidence": round(float(prediction.emotion_confidence), 3),
            "emotion_candidates": list(prediction.emotion_candidates),
            "output": prediction.output,
            "timestamp": prediction.timestamp,
        }


def read_latest_prediction():
    with _prediction_lock:
        return dict(_latest_prediction)


@app.route("/")
def index():
    runtime = get_runtime()
    return render_template(
        "index.html",
        ready=runtime.ready,
        warnings=runtime.warnings,
    )


@app.route("/status")
def status():
    runtime = get_runtime()
    return jsonify(
        {
            "ready": runtime.ready,
            "warnings": runtime.warnings,
        }
    )


@app.route("/prediction")
def prediction():
    runtime = get_runtime()
    payload = read_latest_prediction()
    payload["ready"] = runtime.ready
    payload["warnings"] = runtime.warnings
    return jsonify(payload)


def generate_frames():
    try:
        import cv2
    except Exception:
        return

    runtime = get_runtime()
    if not runtime.ready:
        return

    while True:
        prediction = runtime.read_prediction()
        if prediction is None:
            break

        update_latest_prediction(prediction)
        frame = apply_prediction_overlay(prediction.frame, prediction)
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False,
    )

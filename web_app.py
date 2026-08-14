import os
import threading
import time

# MUST be set before importing TensorFlow or Keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, Response, jsonify, render_template, request

from utils.pipeline import AssistiveCommunicationRuntime, apply_prediction_overlay

app = Flask(__name__)
_runtime = None
_runtime_lock = threading.Lock()
_worker_started = False
_worker_lock = threading.Lock()
_latest_jpeg = None
_jpeg_lock = threading.Lock()
_latest_prediction = {
    "gesture": None,
    "emotion": None,
    "gesture_confidence": 0.0,
    "emotion_confidence": 0.0,
    "emotion_candidates": [],
    "accumulated_sentence": "",
    "interpreted_sentence": "",
    "translated_output": None,
    "output": None,
    "timestamp": None,
}
_prediction_lock = threading.Lock()
_init_thread_started = False
_init_lock = threading.Lock()

_PLACEHOLDER_JPEG = None


def _get_placeholder_jpeg():
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is not None:
        return _PLACEHOLDER_JPEG
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)
    cv2.putText(
        img,
        "Starting camera...",
        (120, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (220, 220, 220),
        2,
    )
    cv2.putText(
        img,
        "Please wait",
        (220, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
    )
    ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        _PLACEHOLDER_JPEG = buffer.tobytes()
    return _PLACEHOLDER_JPEG


def _encode_jpeg(frame):
    try:
        import cv2
    except Exception:
        return None
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        return buffer.tobytes()
    return None


def get_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime

    runtime = AssistiveCommunicationRuntime(
        enable_emotion=False,
        fast_mode=True,
    )

    with _runtime_lock:
        if _runtime is None:
            _runtime = runtime
            start_processing_worker()
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
            "accumulated_sentence": prediction.accumulated_sentence or "",
            "interpreted_sentence": prediction.interpreted_sentence or "",
            "translated_output": prediction.translated_output,
            "output": prediction.output,
            "timestamp": prediction.timestamp,
        }


def read_latest_prediction():
    with _prediction_lock:
        return dict(_latest_prediction)


def _processing_loop():
    """Single loop: capture frame -> stream video -> run sign recognition."""
    global _latest_jpeg, _runtime

    while True:
        with _runtime_lock:
            runtime = _runtime
        if runtime is None or not runtime.ready:
            time.sleep(0.1)
            continue
        break

    while True:
        try:
            result = runtime.video.get_frame()
            if result is None:
                time.sleep(0.05)
                continue

            frame, timestamp = result

            # Stream raw camera immediately (fixes black screen)
            raw_jpeg = _encode_jpeg(frame)
            if raw_jpeg:
                with _jpeg_lock:
                    _latest_jpeg = raw_jpeg

            prediction = runtime.process_frame(frame, timestamp)
            if prediction is not None:
                update_latest_prediction(prediction)
                annotated = apply_prediction_overlay(prediction.frame, prediction)
                ann_jpeg = _encode_jpeg(annotated)
                if ann_jpeg:
                    with _jpeg_lock:
                        _latest_jpeg = ann_jpeg

        except Exception as exc:
            import sys
            sys.stderr.write(f"[web_app] processing error: {exc}\n")
            time.sleep(0.05)


def start_processing_worker():
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_processing_loop, daemon=True)
        thread.start()
        _worker_started = True


def ensure_runtime_init():
    global _init_thread_started
    with _init_lock:
        if _init_thread_started:
            return
        thread = threading.Thread(target=get_runtime, daemon=True)
        thread.start()
        _init_thread_started = True


@app.route("/")
def index():
    ensure_runtime_init()
    with _runtime_lock:
        runtime = _runtime
    warnings = list(runtime.warnings) if runtime is not None else []
    if runtime is None:
        warnings = ["Starting camera and sign model…"] + warnings
    return render_template(
        "index.html",
        ready=True,
        warnings=warnings,
    )


@app.route("/status")
def status():
    ensure_runtime_init()
    with _runtime_lock:
        runtime = _runtime
    if runtime is None:
        return jsonify(
            {
                "ready": False,
                "warnings": ["Starting camera and sign model…"],
                "target_language": "en",
                "tts_enabled": False,
            }
        )
    return jsonify(
        {
            "ready": runtime.ready,
            "warnings": runtime.warnings,
            "target_language": runtime.target_language,
            "tts_enabled": runtime.tts_enabled,
        }
    )


@app.route("/config", methods=["POST"])
def config():
    ensure_runtime_init()
    runtime = get_runtime()
    data = request.json or {}
    if "target_language" in data:
        runtime.target_language = data["target_language"]
    if "tts_enabled" in data:
        runtime.tts_enabled = bool(data["tts_enabled"])

    return jsonify(
        {
            "status": "success",
            "target_language": runtime.target_language,
            "tts_enabled": runtime.tts_enabled,
        }
    )


@app.route("/prediction")
def prediction():
    ensure_runtime_init()
    payload = read_latest_prediction()
    with _runtime_lock:
        runtime = _runtime
    payload["ready"] = runtime.ready if runtime is not None else False
    payload["warnings"] = (
        runtime.warnings if runtime is not None else ["Starting camera and sign model…"]
    )
    return jsonify(payload)


def generate_frames():
    ensure_runtime_init()
    placeholder = _get_placeholder_jpeg()

    while True:
        with _jpeg_lock:
            jpeg = _latest_jpeg
        if not jpeg:
            jpeg = placeholder
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/clear_sentence", methods=["POST"])
def clear_sentence():
    ensure_runtime_init()
    with _runtime_lock:
        runtime = _runtime
    if runtime is None:
        return jsonify({"status": "cleared"})
    if hasattr(runtime, "_accumulator"):
        runtime._accumulator.clear()
    with _prediction_lock:
        _latest_prediction["accumulated_sentence"] = ""
        _latest_prediction["interpreted_sentence"] = ""
        _latest_prediction["translated_output"] = None
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    ensure_runtime_init()
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False,
        threaded=True,
        host="127.0.0.1",
        port=5000,
    )

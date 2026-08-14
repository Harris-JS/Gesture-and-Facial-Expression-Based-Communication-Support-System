import importlib.util
import os
import sys

# Allow running from scripts/ subfolder — add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


REQUIRED_FILES = [
    "app.py",
    "web_app.py",
    "docs/PLAN.md",
    "README.md",
    "utils/pipeline.py",
    "models/gesture/gesture_model.py",
    "models/emotion/emotion_model.py",
    "templates/index.html",
    "static/styles.css",
]

OPTIONAL_MODEL_FILES = [
    "models/gesture/gesture_model.h5",
    "models/emotion/emotion_model.h5",
]

DEPENDENCIES = ["cv2", "mediapipe", "tensorflow", "flask", "numpy"]


def exists(path):
    return os.path.exists(path)


def dependency_status(module_name):
    return importlib.util.find_spec(module_name) is not None


def check_python_version():
    version = sys.version_info
    if (version.major, version.minor) < (3, 10) or (version.major, version.minor) > (3, 12):
        return False, f"Python {version.major}.{version.minor} detected. Recommended version is 3.10-3.12."
    return True, f"Python {version.major}.{version.minor} is within the recommended range."


def check_runtime_probe(runtime_class=None):
    if runtime_class is None:
        from utils.pipeline import AssistiveCommunicationRuntime

        runtime_class = AssistiveCommunicationRuntime

    runtime = None
    try:
        runtime = runtime_class()
    except Exception as exc:
        return False, [f"Desktop runtime failed to initialize: {exc}"]

    try:
        messages = []
        if runtime.ready:
            messages.append("Desktop runtime initialized with camera access.")
        else:
            messages.append("Desktop runtime initialized, but camera access is unavailable.")

        for warning in getattr(runtime, "warnings", []):
            messages.append(warning)

        return True, messages
    finally:
        if runtime is not None and hasattr(runtime, "release"):
            try:
                runtime.release()
            except Exception:
                pass


def check_web_routes(flask_app=None):
    if flask_app is None:
        from web_app import app as flask_app

    try:
        client = flask_app.test_client()
        index_response = client.get("/")
        status_response = client.get("/status")
        prediction_response = client.get("/prediction")
    except Exception as exc:
        return False, [f"Web app failed to serve test requests: {exc}"]

    messages = [
        f"GET / returned {index_response.status_code}.",
        f"GET /status returned {status_response.status_code}.",
        f"GET /prediction returned {prediction_response.status_code}.",
    ]

    if index_response.status_code != 200 or status_response.status_code != 200 or prediction_response.status_code != 200:
        return False, messages

    payload = status_response.get_json(silent=True) or {}
    messages.append(f"/status ready={payload.get('ready')}.")

    warnings = payload.get("warnings") or []
    if warnings:
        messages.append(f"/status warnings={len(warnings)}.")

    return True, messages


def main():
    ok = True
    runtime_ok = True

    py_ok, py_message = check_python_version()
    print(py_message)
    ok = ok and py_ok

    print("\nRequired files")
    for path in REQUIRED_FILES:
        present = exists(path)
        marker = "OK" if present else "MISSING"
        print(f"[{marker}] {path}")
        ok = ok and present

    print("\nDependency availability")
    dependencies_ok = True
    for name in DEPENDENCIES:
        present = dependency_status(name)
        marker = "OK" if present else "MISSING"
        print(f"[{marker}] {name}")
        dependencies_ok = dependencies_ok and present
    ok = ok and dependencies_ok

    print("\nModel files")
    for path in OPTIONAL_MODEL_FILES:
        present = exists(path)
        marker = "OK" if present else "OPTIONAL"
        print(f"[{marker}] {path}")

    print("\nRuntime probe")
    runtime_probe_ok, runtime_messages = check_runtime_probe()
    runtime_marker = "OK" if runtime_probe_ok else "ERROR"
    for message in runtime_messages:
        print(f"[{runtime_marker}] {message}")
    runtime_ok = runtime_ok and runtime_probe_ok

    print("\nWeb routes")
    web_ok, web_messages = check_web_routes()
    web_marker = "OK" if web_ok else "ERROR"
    for message in web_messages:
        print(f"[{web_marker}] {message}")
    runtime_ok = runtime_ok and web_ok

    if ok:
        if runtime_ok:
            print("\nProject verification passed.")
        else:
            print("\nProject verification found runtime blockers.")
            return 1
        if runtime_probe_ok and any("unavailable" in message.lower() for message in runtime_messages):
            print("Runtime note: connect a webcam to enable live capture.")
        return 0

    print("\nProject verification found blockers.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# Assistive Gesture + Emotion Recognition

This project recognizes hand gestures from a webcam, adds facial-expression context, and produces fused text output for assistive communication.

The runtime now:

- scans camera indices `0` to `4` automatically on Windows using the default, DirectShow, and Media Foundation backends,
- falls back to simple detection status like `Hand detected` or `Face detected` when a model is not ready,
- supports a stronger pretrained Hugging Face emotion model with a local CNN fallback.

## Setup With uv

This repo has been installed and tested with `uv 0.11.7` and Python `3.12.13` on Windows.

Use the `uv` command directly for this project. `python -m uv ...` does not work in this setup because `uv` is installed as a CLI tool, not as an importable module inside the virtual environment.

1. Install `uv`:

```bash
python -m pip install uv
```

2. Install Python `3.12` locally:

```bash
uv python install 3.12
```

If `uv` reports that `python3.12.exe` already exists but is not managed by `uv`, rerun:

```bash
uv python install 3.12 --force
```

3. Create and activate the virtual environment:

```bash
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1
```

4. Install the project dependencies:

```bash
uv pip install -r requirements.txt
```

The first run of the emotion recognizer may download the pretrained model `trpakov/vit-face-expression` into your Hugging Face cache.

## Verify The Environment

Check that the repo, imports, runtime wiring, and Flask routes are ready:

```bash
.\.venv\Scripts\python.exe verify_project.py
```

Run the smoke tests:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Probe webcam access directly:

```bash
.\.venv\Scripts\python.exe test_camera.py
```

## Run The Project

Desktop application:

```bash
.\.venv\Scripts\python.exe app.py
```

Browser application:

```bash
.\.venv\Scripts\python.exe web_app.py
```

Then open `http://127.0.0.1:5000`.

## Data Collection And Training

Collect gesture samples:

```bash
.\.venv\Scripts\python.exe utils/data_collection.py
```

Train the gesture model:

```bash
.\.venv\Scripts\python.exe models/gesture/train_model.py
```

Collect emotion samples:

```bash
.\.venv\Scripts\python.exe models/emotion/emotion_data_collection.py --label happy
```

Train the emotion model:

```bash
.\.venv\Scripts\python.exe models/emotion/train_emotion_model.py
```

If `dataset/emotion/emotion_features.csv` does not exist yet, the trainer can build it from `dataset/dataset/FER2013/train`.

## Notes

- Press `q` in OpenCV windows to exit capture.
- Gesture prediction uses `models/gesture/gesture_model.h5` when available.
- Emotion prediction uses `models/emotion/emotion_model.h5` and `models/emotion/labels.txt` when available.
- When `transformers`, `torch`, and the cached Hugging Face model are available, the app prefers the pretrained `trpakov/vit-face-expression` backend for better emotion accuracy.
- The browser status endpoint is available at `/status`.
- A webcam is required for live capture in both the desktop and browser experiences.
- You can override camera selection with `ASSISTIVE_CAMERA_INDEX` and backend selection with `ASSISTIVE_CAMERA_BACKEND`.

## Troubleshooting

- If `verify_project.py` reports a Python version outside `3.10` to `3.12`, recreate the environment with Python `3.12`.
- If `verify_project.py` says the runtime initialized but camera access is unavailable, the code is installed correctly but Windows is not exposing a webcam to the app.
- If the desktop app reports `Camera unavailable`, close other webcam apps and retry. Then test with:

```bash
.\.venv\Scripts\python.exe test_camera.py
```

- If camera index `0` is not the right device, try:

```powershell
$env:ASSISTIVE_CAMERA_INDEX="1"
.\.venv\Scripts\python.exe app.py
```

- If one Windows backend works better on your machine, try:

```powershell
$env:ASSISTIVE_CAMERA_BACKEND="dshow"
.\.venv\Scripts\python.exe app.py
```

- If the browser app is launched with `python web_app.py`, it now starts without Flask's auto-reloader so the webcam is not opened twice.
- If emotion prediction is unavailable, train `models/emotion/emotion_model.h5` first.
- The emotion trainer now works without `pandas`, and the gesture trainer works without `scikit-learn`, so `requirements.txt` is enough for both training scripts.
- The current gesture labels (`Hello`, `Yes`, `No`, `Thank You`, `Help`) are sign words that are often dynamic. For strong accuracy, they need a video-based sign recognition model rather than a single-frame classifier.

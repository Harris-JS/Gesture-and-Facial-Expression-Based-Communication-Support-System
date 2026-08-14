# Assistive Gesture + Emotion Recognition

This project recognizes hand gestures from a webcam, adds facial-expression context, and produces fused text output for assistive communication.

The runtime now:
- scans camera indices `0` to `4` automatically on Windows using the default, DirectShow, and Media Foundation backends,
- falls back to simple detection status like `Hand detected` or `Face detected` when a model is not ready,
- supports a stronger pretrained Hugging Face emotion model with a local CNN fallback.

---

## Project Directory Structure

The project has been cleaned and structured into a highly organized folder layout:

```
project/
├── app.py                      # Desktop entry point
├── web_app.py                  # Flask web application entry point
├── requirements.txt            # Project dependencies
├── pyproject.toml              # Metadata and project description
├── .gitignore                  # Git ignore rules
├── .python-version             # Python version pin (3.12)
├── README.md                   # This README file
│
├── docs/                       # Project documentation and analysis
│   ├── PLAN.md                 # Original development plan
│   ├── PROJECT_ANALYSIS.md     # In-depth architectural analysis
│   └── project_algorithm_and_issues_report.md # Algorithmic report
│
├── scripts/                    # Utility scripts for developers
│   ├── fix_data.py             # Script to normalize training coordinates
│   ├── verify_project.py       # Full-system verification test
│   └── test_camera.py          # Ad-hoc webcam connection testing tool
│
├── models/                     # Deep Learning models and training code
│   ├── gesture/                # Hand gesture recognition pipeline (LSTM)
│   │   ├── gesture_model.py
│   │   ├── gesture_model.h5
│   │   ├── lstm_gesture_model.h5
│   │   ├── lstm_gesture_model.keras
│   │   ├── labels.txt
│   │   ├── train_model.py
│   │   ├── train_sequence_model.py
│   │   └── extract_and_train.py
│   └── emotion/                # Facial emotion recognition pipeline
│       ├── emotion_model.py
│       ├── emotion_model.h5
│       ├── normalization.npz
│       ├── labels.txt
│       ├── emotion_data_collection.py
│       └── train_emotion_model.py
│
├── utils/                      # Helper modules and reusable logic
│   ├── pipeline.py             # Core real-time fusion pipeline
│   ├── input_module.py         # OpenCV and MediaPipe camera feed handler
│   ├── feature_extraction.py   # Coordinate processing and scaling
│   ├── preprocessing.py        # Frame normalization and augmentation
│   ├── data_collection.py      # Script to save raw landmarks
│   ├── translation.py          # Language translation support
│   ├── speech.py               # Text-to-speech integration
│   └── runtime_env.py          # Environment diagnostics
│
├── dataset/                    # Training and feature datasets
│   ├── gesture/                # Gestures, including 'ASL Dataset'
│   │   └── ASL Dataset/
│   ├── emotion/                # Emotion landmark features and datasets
│   │   ├── emotion_features.csv
│   │   └── FER2013/
│   └── sign_language/          # Signs, including 'wlasl_features'
│       └── wlasl_features/
│
├── static/                     # Web assets (CSS)
│   └── styles.css
│
├── templates/                  # Flask templates
│   └── index.html
│
└── tests/                      # Suite of automated tests
    └── test_project_smoke.py
```

---

## Setup and Virtual Environment

This project utilizes a consolidated, robust virtual environment in `tf_venv/` preconfigured with Python `3.12.4` and all required packages (TensorFlow, Keras, MediaPipe, OpenCV, Flask, NumPy).

### Activate the Environment:

Activate the environment using PowerShell:
```powershell
.\tf_venv\Scripts\Activate.ps1
```

Or using Git Bash / Cmd:
```bash
source tf_venv/Scripts/activate
```

---

## Verify The Environment

To check that the files, imports, runtime wiring, and Flask routes are ready, run:

```bash
.\tf_venv\Scripts\python.exe scripts/verify_project.py
```

Run the smoke tests to check the components:

```bash
.\tf_venv\Scripts\python.exe -m unittest discover -s tests -v
```

Probe webcam access directly:

```bash
.\tf_venv\Scripts\python.exe scripts/test_camera.py
```

---

## Run The Project

### Desktop application:

```bash
.\tf_venv\Scripts\python.exe app.py
```

### Browser application:

```bash
.\tf_venv\Scripts\python.exe web_app.py
```

Then open `http://127.0.0.1:5000` in your web browser.

---

## Data Collection And Training

Collect gesture samples:

```bash
.\tf_venv\Scripts\python.exe utils/data_collection.py
```

Train the gesture model:

```bash
.\tf_venv\Scripts\python.exe models/gesture/train_model.py
```

Collect emotion samples:

```bash
.\tf_venv\Scripts\python.exe models/emotion/emotion_data_collection.py --label happy
```

Train the emotion model:

```bash
.\tf_venv\Scripts\python.exe models/emotion/train_emotion_model.py
```

If `dataset/emotion/emotion_features.csv` does not exist yet, the trainer will build it from the FER2013 dataset folders at `dataset/emotion/FER2013/train`.

---

## Notes & Troubleshooting

- **Exiting**: Press `q` in OpenCV windows to exit camera feeds.
- **Model Files**: Gesture prediction defaults to `models/gesture/gesture_model.h5`. Emotion prediction defaults to `models/emotion/emotion_model.h5`.
- **Hugging Face Option**: When `transformers` and `torch` are available, the app prefers the pretrained `trpakov/vit-face-expression` backend for state-of-the-art emotion accuracy.
- **Webcam access**: If `verify_project.py` says the runtime initialized but camera access is unavailable, Windows is not exposing your webcam to the app. Close any other apps using the webcam.
- **Alternative Camera Index**: If the default camera index is incorrect, override it via environment variables:
  ```powershell
  $env:ASSISTIVE_CAMERA_INDEX="1"
  .\tf_venv\Scripts\python.exe app.py
  ```

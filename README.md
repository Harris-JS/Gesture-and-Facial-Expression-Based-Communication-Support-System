# Assistive Gesture + Emotion Recognition

This project recognizes hand gestures from a webcam, adds facial-expression context, and produces fused text output for assistive communication.

The runtime now:
- scans camera indices `0` to `4` automatically on Windows using the default, DirectShow, and Media Foundation backends,
- falls back to simple detection status like `Hand detected` or `Face detected` when a model is not ready,
- supports a stronger pretrained Hugging Face emotion model with a local CNN fallback.

---

## Project Directory Structure

The repository contains the source code, trained model files, utilities, web interface, and tests required for the project.

```text
project/
├── app.py                      # Desktop application entry point
├── web_app.py                  # Flask web application entry point
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
│
├── scripts/                    # Utility and testing scripts
│   ├── fix_data.py             # Data normalization utility
│   ├── verify_project.py       # Full-system verification
│   └── test_camera.py          # Webcam testing utility
│
├── models/                     # Machine learning models and training code
│   ├── gesture/                # Hand gesture recognition pipeline
│   │   ├── gesture_model.py
│   │   ├── gesture_model.h5
│   │   ├── lstm_gesture_model.h5
│   │   ├── lstm_gesture_model.keras
│   │   ├── labels.txt
│   │   ├── train_sequence_model.py
│   │   └── extract_and_train.py
│   │
│   └── emotion/                # Facial emotion recognition pipeline
│       ├── emotion_model.py
│       ├── emotion_model.h5
│       ├── normalization.npz
│       └── labels.txt
│
├── utils/                      # Reusable application modules
│   ├── pipeline.py             # Core real-time fusion pipeline
│   ├── input_module.py         # OpenCV and MediaPipe camera handling
│   ├── feature_extraction.py   # Landmark feature processing
│   ├── preprocessing.py        # Frame preprocessing
│   ├── speech.py               # Text-to-speech integration
│   ├── translation.py          # Translation support
│   └── runtime_env.py          # Environment diagnostics
│
├── static/                     # Web assets
│   └── styles.css
│
├── templates/                  # Flask templates
│   └── index.html
│
└── tests/                      # Automated tests
    └── test_project_smoke.py
```

> **Note:** The dataset, virtual environment, editor configuration, and local documentation are intentionally excluded from this repository.

---

## Dataset

The project uses the **FER2013 (Facial Expression Recognition 2013)** dataset for facial emotion recognition.

The dataset is **not included in this repository** because of its size. It must be obtained separately for local training and placed under the project's ignored `dataset/` directory.

Expected local dataset path for emotion training:

```text
dataset/
└── emotion/
    └── FER2013/
        └── train/
```

The `dataset/` directory is excluded through `.gitignore`, so the dataset will not be uploaded to GitHub.

---

## Setup

Create a Python virtual environment locally. The environment itself is intentionally excluded from GitHub.

### Windows PowerShell

```powershell
python -m venv tf_venv
.\tf_venv\Scripts\Activate.ps1
```

### Git Bash / Command Prompt

```bash
source tf_venv/Scripts/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Verify The Environment

To check that the files, imports, runtime wiring, and Flask routes are ready, run:

```powershell
.\tf_venv\Scripts\python.exe scripts/verify_project.py
```

Run the smoke tests:

```powershell
.\tf_venv\Scripts\python.exe -m unittest discover -s tests -v
```

Probe webcam access directly:

```powershell
.\tf_venv\Scripts\python.exe scripts/test_camera.py
```

---

## Run The Project

### Desktop application

```powershell
.\tf_venv\Scripts\python.exe app.py
```

### Browser application

```powershell
.\tf_venv\Scripts\python.exe web_app.py
```

Then open `http://127.0.0.1:5000` in your web browser.

---

## Data Collection And Training

Collect gesture samples:

```powershell
.\tf_venv\Scripts\python.exe utils/data_collection.py
```

Train the gesture model:

```powershell
.\tf_venv\Scripts\python.exe models/gesture/train_model.py
```

Collect emotion samples:

```powershell
.\tf_venv\Scripts\python.exe models/emotion/emotion_data_collection.py --label happy
```

Train the emotion model:

```powershell
.\tf_venv\Scripts\python.exe models/emotion/train_emotion_model.py
```

If `dataset/emotion/emotion_features.csv` does not exist yet, the trainer can build it from the FER2013 dataset placed under `dataset/emotion/FER2013/train`.

---

## Notes & Troubleshooting

- **Exiting:** Press `q` in OpenCV windows to exit camera feeds.
- **Model files:** Gesture prediction defaults to `models/gesture/gesture_model.h5`. Emotion prediction defaults to `models/emotion/emotion_model.h5`.
- **Hugging Face option:** When `transformers` and `torch` are available, the app can use the pretrained `trpakov/vit-face-expression` backend for emotion recognition.
- **Webcam access:** If `verify_project.py` says the runtime initialized but camera access is unavailable, close other applications using the webcam and check Windows camera permissions.
- **Alternative camera index:** If the default camera index is incorrect, override it with an environment variable:

```powershell
$env:ASSISTIVE_CAMERA_INDEX="1"
.\tf_venv\Scripts\python.exe app.py
```

---

## Repository Exclusions

The following local resources are intentionally excluded from version control:

- `dataset/` — large training datasets
- `tf_venv/` — local Python virtual environment
- `.vscode/` and `.idea/` — editor configuration
- `docs/` — local project documentation

This keeps the public repository focused on the source code and project resources needed to understand and reproduce the system.
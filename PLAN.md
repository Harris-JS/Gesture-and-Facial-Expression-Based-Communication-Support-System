# Project Plan: Gesture and Emotion Recognition System

## Overview

This project aims to build a real-time gesture and emotion recognition system using computer vision. It captures webcam input, processes frames for hand/face detection, extracts features, and predicts gestures/emotions via trained models. The system can run as a standalone app or web interface.

## Implemented Modules

- Input Module: Captures webcam frames.
- Preprocessing: Detects hands, face bounding boxes, and face mesh landmarks with MediaPipe.
- Feature Extraction: Extracts hand and face landmark features.
- Gesture Model: Loads a trained TensorFlow gesture classifier when available.
- Emotion Module: Includes emotion data collection, training, and inference scaffolding.
- Integration Pipeline: Combines gesture and emotion predictions into a fused output string.
- Desktop App: Real-time OpenCV application with optional text-to-speech.
- Web Interface: Flask app with templates, static styling, live stream, and status endpoint.
- Verification And Testing: `verify_project.py` plus smoke tests in `tests/`.
- Dependencies: Listed in `requirements.txt`.

## Remaining Gaps

- Trained Emotion Model File: `models/emotion/emotion_model.h5` still needs to be generated.
- Environment Setup: The current machine is on Python 3.14, while the ML stack is better suited to Python 3.10-3.12.
- Dataset Curation: Gesture and emotion sample quality still determines final accuracy.
- Fusion Logic: Current multimodal fusion is a simple text combination and can be improved further.

## Step-by-Step Implementation Plan

1. Create dataset folder and collect gesture data using data_collection.py.
2. Train/update gesture model if needed.
3. Implement emotion feature extraction (e.g., facial landmarks for emotions).
4. Create emotion data collection script.
5. Train emotion model.
6. Build web interface with Flask (routes for video stream, predictions). Completed.
7. Integrate gesture and emotion predictions in app.py or web app. Completed with shared runtime.
8. Add testing, error handling, and documentation. Completed for smoke testing and setup validation.
9. Deploy and optimize for performance. Pending.

## Dependencies

Install via `pip install -r requirements.txt`.

## Usage

- Run `python verify_project.py` to validate the local environment.
- Run `python test_camera.py` to test webcam access.
- Run `python app.py` for the desktop experience.
- Run `python web_app.py` for browser-based access.

# Project Algorithm Analysis and Detection Issues Report

Based on a detailed review of your project's codebase, `PROJECT_ANALYSIS.md`, and `README.md`, here is a complete breakdown of the algorithms currently driving your assistive communication system, along with the root causes of why the detection is not working properly.

## 1. What Algorithms is the Project Running On?

Your project relies on a pipeline of different algorithms for detection, feature extraction, and classification:

### A. Object Detection & Landmark Tracking (MediaPipe)
- **Algorithm**: MediaPipe uses specialized lightweight **Convolutional Neural Networks (CNNs)** to detect hands and faces in the frame. 
- **Usage**: Located in `utils/preprocessing.py`, it extracts 21 3D landmarks for the hand and a detailed 3D face mesh for the face.

### B. Gesture Recognition (Multi-Layer Perceptron / Dense Network)
- **Algorithm**: The gesture classifier (`models/gesture/gesture_model.py`) is a **Feed-Forward Neural Network (MLP)**. It is NOT a CNN.
- **Architecture**: It takes a 42-dimensional vector (21 hand landmarks × 2 coordinates [X, Y]) and passes it through `Dense` (fully connected) layers (64 neurons -> 32 neurons -> 5 output classes with softmax).
- **Usage**: It classifies single frames into 5 gestures (`Hello`, `Yes`, `No`, `Thank You`, `Help`). 

### C. Emotion Recognition (Vision Transformer & CNN)
You mentioned you thought it needed to run on a CNN. In reality, it supports **both** a Transformer and a CNN:
- **Primary Algorithm (Hugging Face)**: A **Vision Transformer (ViT)** model (`trpakov/vit-face-expression`). Transformers are state-of-the-art and generally outperform CNNs for complex image classification tasks.
- **Fallback Algorithm (Local)**: A **Convolutional Neural Network (CNN)** (`models/emotion/emotion_model.h5`) which processes 48x48 grayscale face images.
- **Usage**: Located in `models/emotion/emotion_model.py`. The system attempts to load the Hugging Face ViT first. If `transformers`/`torch` are unavailable, it defaults to your local CNN.

---

## 2. Why is the Detection Not Working Properly? (The Issues)

After debugging the runtime pipeline, I have found the exact reasons why your model is performing poorly in real-world detection:

### Critical Issue #1: Gesture Features are NOT Normalized (Location Dependency)
In `utils/feature_extraction.py`, the hand features are extracted as **absolute pixel coordinates**:
```python
for lm in hand_landmarks.landmark:
    x = lm.x * w
    y = lm.y * h
    features.append(x)
    features.append(y)
```
**Why this breaks detection**: Because you are using absolute screen pixels (e.g., `X: 450, Y: 300`), the model memorizes *where* the hand was during training, rather than the *shape* of the hand. 
- If you train the model with your hand in the center of the screen, and then try to predict with your hand on the left side of the screen, **the model will fail entirely** because the coordinates are completely different numbers.
- **The Fix**: The coordinates must be normalized. They should be calculated *relative* to the wrist (landmark 0), and scaled so that the size of the hand doesn't matter (distance normalization). If you fix this, you will also need to re-record your gesture dataset and retrain the model.

### Critical Issue #2: Emotion Face Area Threshold is Too Strict
In `utils/pipeline.py` (line 318), there is a hardcoded rule that blocks emotion recognition if your face is not close enough to the camera:
```python
if self.emotion_model is not None and face_area_ratio >= 0.05:
    emotion_scores = self.emotion_model.predict_scores(face_image)
```
**Why this breaks detection**: `0.05` means your face must take up at least 5% of the total webcam frame area. In a standard 640x480 webcam view, that requires a face bounding box of roughly 125x125 pixels. If you are sitting slightly further back from your monitor, the `face_area_ratio` will drop below 0.05, and the emotion model **will silently ignore your face** and refuse to predict an emotion.
- **The Fix**: Lower this threshold (e.g., to `0.01` or `0.02`) so the emotion model triggers even when you aren't leaning close to the camera.

### Issue #3: Single-Frame Gesture Instability
As noted in your `PROJECT_ANALYSIS.md`, the gestures you chose (like "Thank You" or "Help") are inherently *dynamic* sign language motions (they involve moving the hand over time). 
- Your current AI model only looks at a single static snapshot (one frame).
- Even though `utils/pipeline.py` tries to stabilize this by requiring 3 consecutive identical predictions, a static model will always struggle to differentiate dynamic gestures.

---

## 3. Summary & Next Steps

Your project correctly utilizes AI (Transformers, CNNs, and MLPs), but the **data preparation** before feeding it to the AI is flawed. 

**Recommended Action Plan:**
1. **Fix `feature_extraction.py`**: Update `extract_hand_features` to compute coordinates relative to the wrist (landmark `0`), so the hand's position on screen doesn't ruin the prediction.
2. **Retrain Gesture Model**: Once the feature extraction is fixed, re-run `utils/data_collection.py` to get new normalized data, and re-run `models/gesture/train_model.py`.
3. **Fix `pipeline.py`**: Lower the `face_area_ratio >= 0.05` threshold to `0.01` or `0.02` so emotion detection works at a normal desk distance.

Let me know if you would like me to write the code to fix these issues immediately!

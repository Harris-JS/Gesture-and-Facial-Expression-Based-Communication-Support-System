# Assistive Communication Project - Architecture and Flow

## 1. Project Purpose

This project is an assistive communication system that uses a webcam to:

- recognize hand gestures,
- estimate facial emotion,
- fuse both results into a single output message,
- display the prediction on screen,
- optionally speak the result using text-to-speech.

It supports two user experiences:

- `app.py`: a desktop OpenCV application,
- `web_app.py`: a browser-based Flask UI with live video streaming.

## 2. High-Level Architecture

The project is organized into four main areas:

1. **Runtime / application entrypoints**:
   - `app.py`
   - `web_app.py`
   - `main.py`
   - `verify_project.py`

2. **Live capture and prediction pipeline**:
   - `utils/pipeline.py`
   - `utils/input_module.py`
   - `utils/preprocessing.py`
   - `utils/feature_extraction.py`

3. **Machine learning models**:
   - `models/gesture/gesture_model.py`
   - `models/emotion/emotion_model.py`
   - `models/gesture/train_model.py`
   - `models/emotion/train_emotion_model.py`

4. **Data collection**:
   - `utils/data_collection.py`
   - `models/emotion/emotion_data_collection.py`

## 3. What Each File Does

### `app.py`

- Starts a desktop application.
- Opens the webcam and reads frames using `AssistiveCommunicationRuntime`.
- Draws prediction labels on each frame.
- If `pyttsx3` is available, speaks the fused output text.
- Exits when the user presses `q`.

### `web_app.py`

- Creates a Flask web app.
- Serves `templates/index.html` and `static/styles.css`.
- Exposes `/prediction`, `/status`, and `/video_feed` endpoints.
- Streams annotated JPEG frames from the webcam.
- Shares the latest prediction data as JSON for the browser UI.

### `verify_project.py`

- Checks required files and dependencies.
- Verifies Python version compatibility.
- Probes the runtime and Flask routes.
- Reports whether the project is ready to run.

### `main.py`

- A trivial placeholder that prints a greeting.
- Not used in the actual inference or UI flow.

### `utils/pipeline.py`

- Implements the core runtime class: `AssistiveCommunicationRuntime`.
- Builds the full prediction pipeline.
- Manages camera initialization and model loading.
- Reads frames, preprocesses them, extracts features, predicts gesture and emotion, smooths results, and constructs output text.
- Creates overlay annotations for visible debugging.

### `utils/input_module.py`

- Implements `VideoCaptureModule`.
- Scans available webcam indices and backends.
- Tries multiple OpenCV capture backends for reliability.
- Reads frames, flips them horizontally for mirror view, and timestamps them.

### `utils/preprocessing.py`

- Uses MediaPipe for hand and face detection.
- Detects up to one hand and one face per frame.
- Returns:
  - the annotated frame,
  - hand landmarks,
  - face mesh landmarks,
  - face bounding box.
- Draws landmarks and face rectangles on the frame.

### `utils/feature_extraction.py`

- Converts detected landmarks into numeric feature vectors.
- `extract_hand_features()` returns a 42-element vector: 21 hand landmarks × 2 coordinates.
- `extract_face_features()` returns normalized face mesh coordinates for emotion data collection.

### `models/gesture/gesture_model.py`

- Loads a saved TensorFlow Keras gesture classifier from `models/gesture/gesture_model.h5`.
- Defines the gesture labels:
  - `Hello`, `Yes`, `No`, `Thank You`, `Help`.
- Predicts one label and its confidence score from hand features.

### `models/gesture/train_model.py`

- Loads gesture training data from `dataset/gesture_data.csv`.
- Splits the data into train/test sets.
- Builds a simple dense neural network on 42-dimensional input:
  - Dense(64, relu)
  - Dense(32, relu)
  - Dense(5, softmax)
- Trains for 50 epochs and saves the model to `models/gesture/gesture_model.h5`.

### `models/emotion/emotion_model.py`

- Loads an emotion predictor using the best available backend:
  - preferred: Hugging Face `trpakov/vit-face-expression` transformer model,
  - fallback: local TensorFlow model at `models/emotion/emotion_model.h5`.
- If the Hugging Face backend is available, it uses image classification on the aligned face crop.
- If not, it uses a grayscale 48×48 CNN on face crops.
- Supports emotion labels such as:
  - `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

### `models/emotion/emotion_data_collection.py`

- Collects face mesh landmark features from live webcam frames.
- Press `space` to save a sample labeled with the user-provided emotion.
- Records samples to `dataset/emotion/emotion_features.csv`.
- Useful for building a custom face-feature dataset.

### `models/emotion/train_emotion_model.py`

- Trains a CNN on FER2013-like image folders under `dataset/dataset/FER2013/train` and `test`.
- Reads each folder as a class label.
- Creates a TensorFlow model trained on 48×48 grayscale face images.
- Saves the trained model and the label text file.

### `utils/runtime_env.py`

- Sets environment variables for runtime stability.
- Silences TensorFlow and logging warnings.
- Stores matplotlib cache path under `.runtime-cache`.

## 4. Prediction Flow from Start to End

### 4.1 Launching the App

Desktop mode:

- Run `python app.py`.
- `app.py` creates `AssistiveCommunicationRuntime()`.
- It opens the webcam and loads both models.
- Each frame enters a prediction loop.

Browser mode:

- Run `python web_app.py`.
- Flask serves the UI.
- The browser pulls frames from `/video_feed`.
- Predictions are updated live in JSON at `/prediction`.

### 4.2 Frame Processing Steps

For each camera frame:

1. `VideoCaptureModule.get_frame()` reads the frame and flips it horizontally.
2. `PreprocessingModule.process(frame)` runs MediaPipe:
   - hand detection,
   - face detection,
   - face mesh detection.
3. `FeatureExtractionModule.extract_hand_features()` converts hand landmarks into a 42-dimensional vector.
4. `AssistiveCommunicationRuntime._extract_aligned_face()` crops and aligns the detected face region.
5. Gesture prediction runs if hand landmarks exist.
6. Emotion prediction runs if a face crop exists and the face occupies enough of the frame.
7. Results are stabilized and smoothed.
8. A fused output message is generated.
9. The frame is annotated and shown or streamed.

## 5. Detailed Internal Workings

### 5.1 Hand Gesture Recognition

Algorithm:

- MediaPipe detects a hand and returns 21 landmarks on the hand.
- Each landmark has normalized x/y coordinates.
- The app converts those landmarks into real pixel coordinates for the current frame.
- This yields a 42-value feature vector: `[x0, y0, x1, y1, ..., x20, y20]`.
- The gesture model is a small feed-forward neural network that maps the 42 inputs to a probability distribution over 5 gesture classes.

Model architecture (in `models/gesture/train_model.py`):

- Input: 42-dimensional feature vector.
- Dense layer with 64 neurons and ReLU activation.
- Dense layer with 32 neurons and ReLU activation.
- Dense output layer with 5 neurons and softmax activation.

How the prediction works:

- The model computes weighted sums of the 42 input coordinates.
- ReLU layers add non-linear decision boundaries.
- The final softmax layer turns raw scores into probabilities.
- The system selects the index with the highest probability and maps it to a gesture label.
- Confidence is the softmax probability for that selected label.

Example:

- The hand pose for `Hello` is converted into 42 numeric values.
- The model outputs `[0.05, 0.12, 0.03, 0.72, 0.08]`.
- `0.72` is the highest score, so the label becomes `Hello` with confidence `0.72`.

Stability:

- A gesture is accepted only when the same predicted label appears with confidence >= 0.60 for 3 consecutive frames.
- This avoids reacting to a single noisy frame.
- For example, if the model sees `Hello` two frames in a row but `Yes` on the third, it will wait until `Hello` is stable.

### 5.2 Facial Emotion Recognition

Algorithm:

- The app uses MediaPipe face detection and face mesh to find and align a face crop.
- A face crop is extracted from the raw image using the detected bounding box.
- Landmark-based alignment rotates the crop so both eyes are roughly horizontal.
- The aligned face image is then classified.

Backends:

1. **Hugging Face ViT classifier**:
   - Uses `transformers` and `torch` if available.
   - The cropped face is converted to a PIL image.
   - The model outputs logits, which are converted to probabilities with softmax.
   - This backend is preferred because it is a pre-trained face expression model.

2. **Local TensorFlow CNN**:
   - Loads `models/emotion/emotion_model.h5` if the Hugging Face model is unavailable.
   - The face image is converted to grayscale, resized to 48×48, and normalized.
   - The CNN outputs a probability distribution over emotion classes.

Local CNN architecture (trained by `models/emotion/train_emotion_model.py`):

- Input: 48×48 grayscale face image.
- Rescaling layer divides pixel values by 255.
- Random Flip, Random Rotation, and Random Zoom for data augmentation during training.
- Three convolutional blocks, each with:
  - Conv2D,
  - BatchNormalization,
  - Conv2D,
  - BatchNormalization,
  - MaxPooling2D,
  - Dropout.
- A final Conv2D with 256 filters.
- GlobalAveragePooling2D to reduce spatial dimensions.
- Dense layer with 128 neurons.
- Output Dense layer with softmax.

Smoothing:

- The runtime stores the last 5 emotion score vectors.
- It averages them to create a smoother prediction.
- The top emotion is accepted only when it is strong enough.
- Rules:
  - for `neutral`: confidence >= 0.75 and margin >= 0.18,
  - for other emotions: confidence >= 0.50 and margin >= 0.08.
- This avoids rapid flipping between similar faces like `happy` and `surprise`.

Example:

- The face crop is classified as:
  - `angry: 0.10`, `happy: 0.60`, `neutral: 0.20`, others lower.
- The model selects `happy` with confidence `0.60`.
- If the average of recent frames still favors `happy`, the runtime outputs `happy`.

### 5.3 Output Message Fusion

The final message is built by `models/emotion.emotion_model.build_output_message()`:

- If both gesture and emotion are recognized, output is:
  - `Gesture + Emotion`
- If only one is recognized, it returns that label.
- If no label is strong enough, it returns a fallback:
  - `Hand detected`, `Face detected`, or `Hand and face detected`.

Example:

- Detected gesture `Hello` and emotion `happy` → output: `Hello + happy`.
- Detected only a face with no confident emotion → output: `Face detected`.

### 5.4 End-to-End Algorithm Summary

1. **Capture and mirror the frame**
   - The webcam frame is read and flipped for a natural user view.
2. **Detect landmarks**
   - MediaPipe finds the hand and face in the same frame.
3. **Extract features**
   - Hand landmarks become a 42-element coordinate vector.
   - Face landmarks create a face crop and alignment.
4. **Run each model**
   - Gesture model predicts one of 5 hand labels.
   - Emotion model predicts one of 7 facial expressions.
5. **Stabilize prediction results**
   - Gesture needs 3 consistent frames.
   - Emotion averages scores from 5 recent frames.
6. **Fuse the labels**
   - If both are valid, combine them with `+`.
   - Otherwise use the available detection.
7. **Annotate and display**
   - The frame is annotated with gesture, emotion, and output text.
   - The output is streamed or spoken.

## 6. Example Prediction Scenario

### Example: User says hello with a happy face

1. The user shows a hand pose corresponding to `Hello`.
2. The webcam captures a frame.
3. MediaPipe finds the hand landmarks and face region.
4. The gesture model receives 42 hand coordinates and predicts `Hello` with confidence `0.82`.
5. The face crop is passed to the emotion model.
6. The emotion model predicts `happy` with confidence `0.78`.
7. The runtime fuses them into `Hello + happy`.
8. On-screen overlay shows:
   - `Gesture: Hello (0.82)`
   - `Emotion: happy (0.78)`
   - `Output: Hello + happy`
9. If available, the desktop app also speaks `Hello + happy`.

### Example: face only, no gesture

1. The user looks at the camera with a neutral face.
2. Hand detection may return no hand landmarks.
3. The emotion model predicts `neutral`.
4. Output becomes `neutral`.
5. If face detection is present but emotion is not stable, the fallback is `Face detected`.

## 7. Training and Data Collection

### Gesture data collection

- Run `python utils/data_collection.py`.
- Enter a label in `0-4`.
- Press `q` to stop recording.
- The script appends `dataset/gesture_data.csv` with rows of format:
  - `[label, x0, y0, x1, y1, ..., x20, y20]`
- Labels map to the gestures in `models/gesture/gesture_model.py`.

### Gesture training

- Run `python models/gesture/train_model.py`.
- It trains the small dense network on `dataset/gesture_data.csv`.
- Saves `models/gesture/gesture_model.h5`.

### Emotion data collection

- Run `python models/emotion/emotion_data_collection.py --label happy`.
- Press `space` to save a sample once the face is detected.
- Uses `utils/feature_extraction.extract_face_features()` to save face landmark coordinates.

### Emotion training

- Run `python models/emotion/train_emotion_model.py`.
- It expects data under `dataset/dataset/FER2013/train` and `dataset/dataset/FER2013/test`.
- It saves the trained model to `models/emotion/emotion_model.h5` and labels to `models/emotion/labels.txt`.

## 8. Dataset Layout

Important data folders:

- `dataset/gesture_data.csv` - collected hand gesture training data.
- `dataset/emotion/emotion_features.csv` - collected face feature samples.
- `dataset/dataset/FER2013/train/` - emotion training images organized by class.
- `dataset/dataset/FER2013/test/` - emotion test images organized by class.

## 9. How the System Predicts

1. **Capture frame**
   - Webcam frame is obtained and mirrored for the user.

2. **Detect landmarks**
   - `mediapipe` finds hand and face landmarks.

3. **Extract features**
   - Hand landmarks → 42 raw coordinates.
   - Face crop → aligned face image.

4. **Run models**
   - Gesture model classifies hand pose.
   - Emotion model classifies face expression.

5. **Smooth and stabilize**
   - Gesture requires 3 consistent frames.
   - Emotion averages the previous 5 score vectors.

6. **Build output**
   - Fused message is created.
   - The frame is annotated for display.

7. **Display / stream**
   - The output is shown in OpenCV or streamed through Flask.

## 10. Practical Notes

- The system is frame-based, not video-sequence based.
- Gesture recognition may fail if the hand is occluded, not fully visible, or not detected clearly.
- Emotion detection depends on a visible face and reasonable alignment.
- `App.py` speaks only if `pyttsx3` is installed.
- The project prefers the Hugging Face emotion model but falls back to the local CNN.
- `verify_project.py` is useful to confirm environment readiness before running.

## 11. Commands to Run

Desktop app:

```bash
python app.py
```

Web app:

```bash
python web_app.py
```

Verify project:

```bash
python verify_project.py
```

Collect gesture data:

```bash
python utils/data_collection.py
```

Train gesture model:

```bash
python models/gesture/train_model.py
```

Train emotion model:

```bash
python models/emotion/train_emotion_model.py
```

## 12. Summary

This project is an assistive live prediction system that combines:

- MediaPipe landmark detection,
- a simple hand gesture classifier,
- a facial emotion classifier,
- temporal smoothing,
- and UI output via OpenCV or Flask.

It is designed to be extendable, with clear entry points for data collection, training, runtime inference, and web deployment.

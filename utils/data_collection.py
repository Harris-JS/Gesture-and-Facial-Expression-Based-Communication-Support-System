import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from utils.input_module import VideoCaptureModule
from utils.preprocessing import PreprocessingModule
from utils.feature_extraction import FeatureExtractionModule


video = VideoCaptureModule()
preprocess = PreprocessingModule()
feature_extractor = FeatureExtractionModule()

label = int(input("Enter gesture label (0-4): "))
os.makedirs("dataset", exist_ok=True)

with open("dataset/gesture_data.csv", mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    while True:
        result = video.get_frame()
        if result is None:
            break

        frame, _ = result
        frame, hand_landmarks, _, _ = preprocess.process(frame)
        features = feature_extractor.extract_hand_features(hand_landmarks, frame.shape)

        if features is not None:
            row = [label] + features.tolist()
            writer.writerow(row)

        cv2.putText(
            frame,
            f"Recording Label: {label}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()

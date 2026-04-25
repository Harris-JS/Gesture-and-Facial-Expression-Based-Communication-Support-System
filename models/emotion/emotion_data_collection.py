"""Collect face landmark features and save to CSV with labels.

Usage:
 python emotion_data_collection.py --label happy

Press 'q' to quit; space to record current frame sample.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.input_module import VideoCaptureModule
from utils.preprocessing import PreprocessingModule
from utils.feature_extraction import FeatureExtractionModule

def main(label, out_dir="dataset/emotion"):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "emotion_features.csv")

    video = VideoCaptureModule()
    preprocess = PreprocessingModule()
    fe = FeatureExtractionModule()

    print("Press SPACE to save a sample for label:", label)

    with open(out_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        while True:
            res = video.get_frame()
            if res is None:
                break
            frame, _ = res
            frame, hand_landmarks, face_landmarks, face_bbox = preprocess.process(frame)

            cv2_frame = frame
            cv2.imshow("Emotion data collection", cv2_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:  # space
                feats = fe.extract_face_features(face_landmarks, frame.shape)
                if feats is None:
                    print("No face landmarks detected, sample skipped")
                    continue
                row = list(feats.tolist())
                row.append(label)
                writer.writerow(row)
                f.flush()
                print("Saved sample, length:", len(row)-1)

    video.release()

if __name__ == '__main__':
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    main(args.label)

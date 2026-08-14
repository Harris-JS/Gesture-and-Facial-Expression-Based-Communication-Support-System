"""
Record 30-frame sign sequences from your webcam for custom demo words
(hello, hi, welcome, thanks, please).

Run from project root:
    .\\tf_venv\\Scripts\\python.exe models/gesture/collect_demo_sequences.py

Controls
--------
  SPACE  — save one 30-frame sample for the current word
  N      — next word in the list
  P      — previous word
  Q      — quit
"""

import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gesture.demo_vocabulary import CUSTOM_DEMO_WORDS, SIGN_GUIDE
from utils.feature_extraction import FeatureExtractionModule
from utils.input_module import VideoCaptureModule
from utils.preprocessing import PreprocessingModule
from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

SEQUENCE_LENGTH = 30
FEATURES = 225
OUT_ROOT = PROJECT_ROOT / "dataset" / "sign_language" / "demo_features" / "sequences"


def next_sample_path(word_dir: Path) -> Path:
    word_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in word_dir.glob("*.npy")]
    return word_dir / f"sample_{len(existing):03d}.npy"


def main():
    if not CUSTOM_DEMO_WORDS:
        print("No custom words defined.")
        return 1

    word_index = 0
    buffer = deque(maxlen=SEQUENCE_LENGTH)

    video = VideoCaptureModule(max_camera_index=1, backend="DSHOW" if os.name == "nt" else None)
    preprocess = PreprocessingModule(fast_mode=True)
    extractor = FeatureExtractionModule()

    print("=" * 60)
    print("  Demo sign collection — custom words")
    print("=" * 60)
    print("Record at least 8–10 samples per word for a usable demo.")
    print()

    try:
        while True:
            word = CUSTOM_DEMO_WORDS[word_index]
            guide = SIGN_GUIDE.get(word, "Perform the sign clearly in front of the camera.")

            result = video.get_frame()
            if result is None:
                print("Camera frame unavailable.")
                break

            frame, _ = result
            frame, holistic, _, _ = preprocess.process(frame)

            if holistic is not None:
                feats = extractor.extract_holistic_features(
                    holistic.pose_landmarks,
                    holistic.left_hand_landmarks,
                    holistic.right_hand_landmarks,
                )
                buffer.append(feats)
            else:
                buffer.append(np.zeros(FEATURES, dtype=np.float32))

            samples = len(list((OUT_ROOT / word).glob("*.npy"))) if (OUT_ROOT / word).exists() else 0

            cv2.putText(frame, f"Word: {word.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Samples saved: {samples}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "SPACE=save  N=next  P=prev  Q=quit", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

            y = 190
            for line in _wrap_text(guide, 50):
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                y += 28

            cv2.imshow("Collect Demo Signs", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("n"):
                word_index = (word_index + 1) % len(CUSTOM_DEMO_WORDS)
                buffer.clear()
                continue
            if key == ord("p"):
                word_index = (word_index - 1) % len(CUSTOM_DEMO_WORDS)
                buffer.clear()
                continue
            if key == ord(" ") and len(buffer) == SEQUENCE_LENGTH:
                seq = np.array(list(buffer), dtype=np.float32)
                out_path = next_sample_path(OUT_ROOT / word)
                np.save(out_path, seq)
                print(f"Saved {out_path}")
                buffer.clear()

    finally:
        video.release()
        cv2.destroyAllWindows()

    print(f"\nSamples stored under: {OUT_ROOT}")
    return 0


def _wrap_text(text: str, width: int):
    words = text.split()
    lines, current = [], []
    for w in words:
        trial = " ".join(current + [w])
        if len(trial) <= width:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


if __name__ == "__main__":
    raise SystemExit(main())

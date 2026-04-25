import numpy as np

class FeatureExtractionModule:

    def extract_hand_features(self, hand_landmarks, frame_shape):
        if hand_landmarks is None:
            return None

        h, w, _ = frame_shape
        features = []

        for lm in hand_landmarks.landmark:
            x = lm.x * w
            y = lm.y * h
            features.append(x)
            features.append(y)

        return np.array(features)  # 42 values (21 points * 2)

    def extract_face_features(self, face_landmarks, frame_shape):
        """Extract flattened normalized (x,y) coordinates from face landmarks.

        Returns a 1D numpy array of length (N_landmarks*2) or None.
        """
        if face_landmarks is None:
            return None

        features = []

        for lm in face_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)

        return np.array(features)

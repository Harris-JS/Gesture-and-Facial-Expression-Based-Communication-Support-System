import numpy as np

class FeatureExtractionModule:
    def extract_holistic_features(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        """
        Extracts 225 features matching the WLASL dataset format:
        Pose (33*3 = 99) + Left Hand (21*3 = 63) + Right Hand (21*3 = 63) = 225
        """
        # Pose: 33 landmarks
        if pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 3)

        # Left Hand: 21 landmarks
        if left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21 * 3)

        # Right Hand: 21 landmarks
        if right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21 * 3)

        return np.concatenate([pose, lh, rh])

    def extract_hand_features(self, hand_landmarks, frame_shape):
        """Legacy static hand features (42)"""
        if hand_landmarks is None:
            return None

        h, w, _ = frame_shape
        features = []
        base_x = hand_landmarks.landmark[0].x * w
        base_y = hand_landmarks.landmark[0].y * h

        coords = []
        for lm in hand_landmarks.landmark:
            x = (lm.x * w) - base_x
            y = (lm.y * h) - base_y
            coords.append([x, y])

        max_val = np.max(np.abs(coords))
        if max_val == 0:
            max_val = 1.0

        for point in coords:
            features.append(point[0] / max_val)
            features.append(point[1] / max_val)

        return np.array(features)

    def extract_face_features(self, face_landmarks, frame_shape):
        if face_landmarks is None:
            return None
        features = []
        for lm in face_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)
        return np.array(features)

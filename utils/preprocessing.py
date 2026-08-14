from utils.runtime_env import configure_runtime_environment
configure_runtime_environment()

import cv2
import mediapipe as mp

class PreprocessingModule:
    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode
        # We will use MediaPipe Holistic to extract everything at once
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0 if fast_mode else 1,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.mp_face = None
        self.face = None
        if not fast_mode:
            self.mp_face = mp.solutions.face_detection
            self.face = self.mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.7,
            )

    @staticmethod
    def _bbox_from_detection(detection, frame_shape):
        h, w, _ = frame_shape
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        return (x, y, bw, bh)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Holistic processing for Gestures (Pose, LH, RH)
        holistic_results = self.holistic.process(rgb)

        # 2. Face detection for Emotion cropping (skipped in fast/web mode)
        face_results = None
        if self.face is not None:
            face_results = self.face.process(rgb)

        face_bbox = None
        if face_results and face_results.detections:
            face_bbox = self._bbox_from_detection(face_results.detections[0], frame.shape)
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        # Draw holistic landmarks for visualization
        if holistic_results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, holistic_results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if holistic_results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, holistic_results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if holistic_results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, holistic_results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        # To keep compatibility with pipeline, we map holistic face landmarks to face_landmarks
        # though it's technically dense face mesh.
        face_landmarks = holistic_results.face_landmarks

        return frame, holistic_results, face_landmarks, face_bbox

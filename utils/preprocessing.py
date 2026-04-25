from utils.runtime_env import configure_runtime_environment

configure_runtime_environment()

import cv2
import mediapipe as mp


class PreprocessingModule:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7,
        )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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

    @staticmethod
    def _bbox_from_face_mesh(face_landmarks, frame_shape, padding_ratio=0.18):
        h, w, _ = frame_shape
        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]
        if not xs or not ys:
            return None

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        size = max(width, height)
        pad = size * padding_ratio

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        half = (size / 2.0) + pad

        x0 = max(0, int(cx - half))
        y0 = max(0, int(cy - half))
        x1 = min(w, int(cx + half))
        y1 = min(h, int(cy + half))

        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(rgb)
        face_results = self.face.process(rgb)
        face_mesh_results = self.face_mesh.process(rgb)

        hand_landmarks = None
        face_landmarks = None
        face_bbox = None

        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
            )

        if face_results.detections:
            face_bbox = self._bbox_from_detection(face_results.detections[0], frame.shape)

        if face_mesh_results and face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            mesh_bbox = self._bbox_from_face_mesh(face_landmarks, frame.shape)
            if mesh_bbox is not None:
                face_bbox = mesh_bbox

        if face_bbox is not None:
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        return frame, hand_landmarks, face_landmarks, face_bbox

import os
import cv2
import time


def _parse_backend(name):
    if not name:
        return None, "default"

    normalized = str(name).strip().upper()
    if normalized in {"DEFAULT", "AUTO"}:
        return None, "default"

    backend_map = {
        "DSHOW": getattr(cv2, "CAP_DSHOW", None),
        "DIRECTSHOW": getattr(cv2, "CAP_DSHOW", None),
        "MSMF": getattr(cv2, "CAP_MSMF", None),
        "MEDIAFOUNDATION": getattr(cv2, "CAP_MSMF", None),
        "V4L": getattr(cv2, "CAP_V4L", None),
        "V4L2": getattr(cv2, "CAP_V4L2", None),
    }
    code = backend_map.get(normalized)
    if code is None:
        return None, normalized.lower()
    return code, normalized.lower()


class VideoCaptureModule:
    def __init__(self, camera_index=0, backend=None, max_camera_index=4):
        self.camera_index = None
        self.backend_name = "default"
        self.cap = None

        previous_log_level = None
        if hasattr(cv2, "setLogLevel") and hasattr(cv2, "getLogLevel"):
            try:
                previous_log_level = cv2.getLogLevel()
                cv2.setLogLevel(0)
            except Exception:
                previous_log_level = None

        requested_index = self._resolve_camera_index(camera_index)
        requested_backend, requested_backend_name = _parse_backend(
            backend or os.environ.get("ASSISTIVE_CAMERA_BACKEND")
        )
        candidates = self._build_candidates(
            requested_index=requested_index,
            requested_backend=requested_backend,
            requested_backend_name=requested_backend_name,
            max_camera_index=max_camera_index,
        )

        errors = []
        try:
            for index, backend_code, backend_name in candidates:
                cap = self._open_capture(index=index, backend_code=backend_code)
                if cap is None:
                    errors.append(f"camera {index} ({backend_name}) could not be created")
                    continue

                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    self.camera_index = index
                    self.backend_name = backend_name
                    break

                cap.release()
                errors.append(f"camera {index} ({backend_name}) unavailable")
        finally:
            if previous_log_level is not None:
                try:
                    cv2.setLogLevel(previous_log_level)
                except Exception:
                    pass

        if self.cap is None:
            tried = ", ".join(errors) if errors else "no camera candidates were available"
            raise Exception(f"Error: Cannot access webcam. Tried {tried}")

    @staticmethod
    def _resolve_camera_index(camera_index):
        env_value = os.environ.get("ASSISTIVE_CAMERA_INDEX")
        candidate = env_value if env_value not in {None, ""} else camera_index
        try:
            return int(candidate)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _open_capture(index, backend_code):
        try:
            if backend_code is None:
                return cv2.VideoCapture(index)
            return cv2.VideoCapture(index, backend_code)
        except Exception:
            return None

    @staticmethod
    def _build_candidates(requested_index, requested_backend, requested_backend_name, max_camera_index):
        backend_candidates = [(requested_backend, requested_backend_name)]

        if os.name == "nt":
            backend_candidates.extend(
                [
                    (getattr(cv2, "CAP_DSHOW", None), "dshow"),
                    (getattr(cv2, "CAP_MSMF", None), "msmf"),
                    (None, "default"),
                ]
            )
        else:
            backend_candidates.extend(
                [
                    (getattr(cv2, "CAP_V4L2", None), "v4l2"),
                    (getattr(cv2, "CAP_V4L", None), "v4l"),
                    (None, "default"),
                ]
            )

        seen = set()
        ordered_indices = [requested_index]
        ordered_indices.extend(index for index in range(max_camera_index + 1) if index != requested_index)

        candidates = []
        for index in ordered_indices:
            for backend_code, backend_name in backend_candidates:
                key = (index, backend_code, backend_name)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((index, backend_code, backend_name))
        return candidates

    def get_frame(self):
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Add timestamp (important for future)
        timestamp = time.time()

        return frame, timestamp

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

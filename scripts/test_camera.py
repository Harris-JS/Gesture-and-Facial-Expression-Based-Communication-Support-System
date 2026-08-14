import cv2

backend_options = [
    ("default", None),
    ("dshow", getattr(cv2, "CAP_DSHOW", None)),
    ("msmf", getattr(cv2, "CAP_MSMF", None)),
]

opened = None
for index in range(5):
    for backend_name, backend_code in backend_options:
        try:
            cap = cv2.VideoCapture(index) if backend_code is None else cv2.VideoCapture(index, backend_code)
        except Exception:
            cap = None

        if cap is not None and cap.isOpened():
            opened = (index, backend_name, cap)
            print(f"Opened camera index {index} with backend {backend_name}")
            break

        if cap is not None:
            cap.release()
    if opened is not None:
        break

if opened is None:
    print("No webcam could be opened on indices 0-4.")
    raise SystemExit(1)

_, _, cap = opened

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

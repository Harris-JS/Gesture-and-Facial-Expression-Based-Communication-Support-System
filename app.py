from utils.pipeline import AssistiveCommunicationRuntime, apply_prediction_overlay


def main():
    try:
        import cv2
    except Exception as exc:
        print(f"OpenCV is required to run the desktop app: {exc}")
        return 1

    runtime = AssistiveCommunicationRuntime()
    if not runtime.ready:
        print("Desktop app could not start.")
        for warning in runtime.warnings:
            print(f"- {warning}")
        return 1

    try:
        import pyttsx3

        tts = pyttsx3.init()
    except Exception:
        tts = None

    if runtime.warnings:
        print("Runtime warnings:")
        for warning in runtime.warnings:
            print(f"- {warning}")

    last_spoken = None

    try:
        while True:
            prediction = runtime.read_prediction()
            if prediction is None:
                break

            frame = apply_prediction_overlay(prediction.frame, prediction)
            cv2.imshow("Assistive Communication", frame)

            if prediction.output and tts is not None and prediction.output != last_spoken:
                try:
                    tts.say(prediction.output)
                    tts.runAndWait()
                    last_spoken = prediction.output
                except Exception:
                    pass

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        runtime.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

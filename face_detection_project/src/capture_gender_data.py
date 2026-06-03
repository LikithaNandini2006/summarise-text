import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
CASCADE_PATH = PROJECT_ROOT / "haarcascade" / "haarcascade_frontalface_default.xml"
CAPTURE_LIMIT = 50


def get_gender():
    gender = input("Enter gender class for training (male/female): ").strip().lower()
    if gender not in {"male", "female"}:
        raise ValueError("Gender class must be either 'male' or 'female'.")
    return gender


def main():
    gender = get_gender()
    output_dir = DATASET_DIR / gender
    output_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    count = 0
    print(f"Capturing {CAPTURE_LIMIT} labeled {gender} images. Press Esc to stop early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            filename = output_dir / f"{gender}_{int(time.time() * 1000)}_{count}.jpg"
            cv2.imwrite(str(filename), face)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{gender}: {count}/{CAPTURE_LIMIT}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            if count >= CAPTURE_LIMIT:
                break

        cv2.imshow("Capture Gender Training Data", frame)
        if count >= CAPTURE_LIMIT or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} labeled images to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

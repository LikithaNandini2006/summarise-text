import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURED_FACES_DIR = PROJECT_ROOT / "captured_faces"
CASCADE_PATH = PROJECT_ROOT / "haarcascade" / "haarcascade_frontalface_default.xml"
CAPTURE_LIMIT = 50


def get_person_name():
    name = input("Enter name: ").strip()
    if not name:
        raise ValueError("Name cannot be empty.")

    safe_name = "".join(char if char.isalnum() else "_" for char in name)
    if not safe_name:
        raise ValueError("Name must contain at least one letter or number.")
    return name, safe_name


def capture_faces(name, safe_name):
    output_dir = CAPTURED_FACES_DIR / safe_name
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(
            f"Cannot save images because {output_dir} is a file. Delete that file "
            "and run capture.py again."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    count = 0
    print(f"Capturing {CAPTURE_LIMIT} face images. Press Esc to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            filename = output_dir / f"{safe_name}_{int(time.time() * 1000)}_{count}.jpg"
            cv2.imwrite(str(filename), face)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name}: {count}/{CAPTURE_LIMIT}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            if count >= CAPTURE_LIMIT:
                break

        cv2.imshow("Capture", frame)

        if count >= CAPTURE_LIMIT or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to {output_dir}")
    return count


def main():
    name, safe_name = get_person_name()
    capture_faces(name, safe_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

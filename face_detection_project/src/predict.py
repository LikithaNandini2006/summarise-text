import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "face_recognition_model.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "face_recognition_labels.json"
GENDER_MODEL_PATH = PROJECT_ROOT / "models" / "gender_model.h5"
CASCADE_PATH = PROJECT_ROOT / "haarcascade" / "haarcascade_frontalface_default.xml"
UNKNOWN_THRESHOLD = 60.0


def load_face_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_model.py before prediction."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Label map not found: {LABELS_PATH}. Run train_model.py before prediction."
        )

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is installed incorrectly or cannot load its DLL files. "
            "Install/fix TensorFlow in your active venv before running prediction."
        ) from exc

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return load_model(MODEL_PATH), labels


def load_gender_model():
    if not GENDER_MODEL_PATH.exists() or GENDER_MODEL_PATH.stat().st_size == 0:
        return None

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is installed incorrectly or cannot load its DLL files. "
            "Install/fix TensorFlow in your active venv before running prediction."
        ) from exc

    return load_model(GENDER_MODEL_PATH)


def load_face_cascade():
    if not CASCADE_PATH.exists() or CASCADE_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Haar cascade not found or empty: {CASCADE_PATH}")

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")

    return cascade


def recognize_face(model, labels, face):
    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    predictions = model.predict(face, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index]) * 100
    name = labels[str(class_index)]
    if confidence < UNKNOWN_THRESHOLD:
        name = "Unknown"
    return name, confidence


def predict_gender(model, face):
    if model is None:
        return "Unavailable", 0.0

    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    prediction = float(model.predict(face, verbose=0)[0][0])
    if prediction > 0.5:
        return "Male", prediction * 100
    return "Female", (1 - prediction) * 100


def draw_prediction(frame, x, y, w, h, name, gender, name_accuracy, gender_accuracy):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    lines = [
        f"Name: {name}",
        f"Gender: {gender}",
        f"Name Accuracy: {name_accuracy:.2f}%",
        f"Gender Accuracy: {gender_accuracy:.2f}%",
    ]
    text_y = max(y - 85, 25)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, text_y + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


def main():
    model, labels = load_face_model()
    gender_model = load_gender_model()
    face_cascade = load_face_cascade()

    if gender_model is None:
        print(
            "Warning: Gender model is unavailable. Add labeled male and female "
            "training images and train a gender model to enable gender prediction."
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Recognizing faces. Press Esc to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            name, name_accuracy = recognize_face(model, labels, face)
            gender, gender_accuracy = predict_gender(gender_model, face)
            draw_prediction(
                frame,
                x,
                y,
                w,
                h,
                name,
                gender,
                name_accuracy,
                gender_accuracy,
            )

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

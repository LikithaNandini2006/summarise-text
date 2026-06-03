import os
import sys
from pathlib import Path


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from capture import capture_faces, get_person_name
from predict import main as recognize_faces
from train_gender_model import main as train_gender_model
from train_model import main as train_model


def main():
    name, safe_name = get_person_name()

    captured_count = capture_faces(name, safe_name)
    if captured_count == 0:
        raise RuntimeError("No face images were captured. Please try again.")

    print("Training face recognition model...")
    train_model()

    print("Training gender model...")
    train_gender_model()

    print("Starting face recognition...")
    recognize_faces()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

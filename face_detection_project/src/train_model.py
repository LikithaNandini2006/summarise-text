import json
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "captured_faces"
MODEL_PATH = PROJECT_ROOT / "models" / "face_recognition_model.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "face_recognition_labels.json"
PROFILES_PATH = PROJECT_ROOT / "models" / "person_profiles.json"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def update_profiles(class_names):
    profiles = {}
    if PROFILES_PATH.exists():
        profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))

    for name in class_names:
        profiles.setdefault(name, {"gender": "Unknown"})

    PROFILES_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def import_tensorflow_tools():
    try:
        from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is installed incorrectly or cannot load its DLL files. "
            "Install/fix TensorFlow in your active venv before training."
        ) from exc

    return Sequential, Input, Conv2D, MaxPooling2D, Flatten, Dense, ImageDataGenerator


def get_usable_class_dirs():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Captured faces folder not found: {DATASET_DIR}")

    usable_class_dirs = []
    skipped_class_dirs = []
    for class_dir in DATASET_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        image_count = sum(
            1
            for image_path in class_dir.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count >= 2:
            usable_class_dirs.append(class_dir)
        else:
            skipped_class_dirs.append(class_dir.name)

    if skipped_class_dirs:
        print(
            "Skipping incomplete capture folders: "
            f"{', '.join(sorted(skipped_class_dirs))}"
        )

    if len(usable_class_dirs) < 2:
        raise ValueError(
            "Capture at least two people before training. Run capture.py once "
            "for each person's name."
        )

    return usable_class_dirs


def create_training_view(class_dirs, view_dir):
    for class_dir in class_dirs:
        destination = view_dir / class_dir.name
        try:
            destination.symlink_to(class_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(class_dir, destination)


def main():
    class_dirs = get_usable_class_dirs()
    (
        Sequential,
        Input,
        Conv2D,
        MaxPooling2D,
        Flatten,
        Dense,
        ImageDataGenerator,
    ) = import_tensorflow_tools()

    with tempfile.TemporaryDirectory(prefix="face_training_") as temp_dir:
        training_view = Path(temp_dir)
        create_training_view(class_dirs, training_view)

        image_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
        common_args = {
            "directory": training_view,
            "target_size": (64, 64),
            "batch_size": 16,
            "class_mode": "categorical",
            "seed": 42,
        }
        train_data = image_gen.flow_from_directory(subset="training", **common_args)
        val_data = image_gen.flow_from_directory(subset="validation", **common_args)

        model = Sequential(
            [
                Input(shape=(64, 64, 3)),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D(),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(train_data.num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(train_data, validation_data=val_data, epochs=10)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)
        labels = {str(index): name for name, index in train_data.class_indices.items()}
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    update_profiles(labels.values())
    print(f"Model saved successfully: {MODEL_PATH}")
    print(f"Labels saved successfully: {LABELS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

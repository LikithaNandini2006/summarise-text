import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "gender_model.h5"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
REQUIRED_CLASSES = {"female", "male"}


def import_tensorflow_tools():
    try:
        from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is installed incorrectly or cannot load its DLL files."
        ) from exc
    return Sequential, Input, Conv2D, MaxPooling2D, Flatten, Dense, ImageDataGenerator


def validate_dataset():
    class_dirs = {path.name.lower(): path for path in DATASET_DIR.iterdir() if path.is_dir()}
    missing_classes = sorted(REQUIRED_CLASSES - class_dirs.keys())
    if missing_classes:
        raise ValueError(
            "Capture labeled gender data first. Missing folders: "
            f"{', '.join(missing_classes)}."
        )

    for class_name in sorted(REQUIRED_CLASSES):
        image_count = sum(
            1
            for image_path in class_dirs[class_name].rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count < 5:
            raise ValueError(
                f"Capture at least 5 labeled images for '{class_name}' before training."
            )


def main():
    validate_dataset()
    (
        Sequential,
        Input,
        Conv2D,
        MaxPooling2D,
        Flatten,
        Dense,
        ImageDataGenerator,
    ) = import_tensorflow_tools()

    image_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    common_args = {
        "directory": DATASET_DIR,
        "target_size": (64, 64),
        "batch_size": 16,
        "class_mode": "binary",
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
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=10)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Gender model saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

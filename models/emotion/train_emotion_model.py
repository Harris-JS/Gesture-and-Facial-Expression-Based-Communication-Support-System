import os
import sys
from pathlib import Path

from utils.runtime_env import configure_runtime_environment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

configure_runtime_environment()

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None


IMAGE_DATASET_ROOT = "dataset/dataset/FER2013/train"
TEST_DATASET_ROOT = "dataset/dataset/FER2013/test"
MODEL_OUT = "models/emotion/emotion_model.h5"
LABELS_OUT = "models/emotion/labels.txt"
IMAGE_SIZE = (48, 48)


def count_images_by_label(root):
    counts = {}
    for label_dir in sorted(Path(root).iterdir()):
        if label_dir.is_dir():
            counts[label_dir.name] = len(list(label_dir.glob("*.jpg")))
    return counts


def discover_classes(root):
    return sorted([path.name for path in Path(root).iterdir() if path.is_dir()])


def collect_image_paths(root, class_names):
    image_paths = []
    labels = []
    for index, label in enumerate(class_names):
        label_dir = Path(root) / label
        for image_path in sorted(label_dir.glob("*.jpg")):
            image_paths.append(str(image_path))
            labels.append(index)
    return image_paths, labels


def split_paths(image_paths, labels, validation_ratio=0.1):
    grouped = {}
    for path, label in zip(image_paths, labels):
        grouped.setdefault(label, []).append(path)

    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []

    for label, paths in grouped.items():
        split_at = max(1, int(len(paths) * (1 - validation_ratio)))
        train_chunk = paths[:split_at]
        val_chunk = paths[split_at:] or paths[:1]
        train_paths.extend(train_chunk)
        train_labels.extend([label] * len(train_chunk))
        val_paths.extend(val_chunk)
        val_labels.extend([label] * len(val_chunk))

    return train_paths, train_labels, val_paths, val_labels


def decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset_from_paths(image_paths, labels, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(len(image_paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size)


def build_model(num_classes):
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.08)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs)


def build_class_weights(class_names, dataset_root):
    counts = count_images_by_label(dataset_root)
    total = sum(counts.values())
    num_classes = len(class_names)
    weights = {}
    for index, label in enumerate(class_names):
        count = max(1, counts.get(label, 1))
        weights[index] = total / (num_classes * count)
    return weights


def build_and_train(epochs=12, batch_size=64):
    if tf is None:
        raise RuntimeError("TensorFlow is not available")

    train_root = Path(IMAGE_DATASET_ROOT)
    test_root = Path(TEST_DATASET_ROOT)
    if not train_root.exists():
        raise FileNotFoundError(train_root)
    if not test_root.exists():
        raise FileNotFoundError(test_root)

    class_names = discover_classes(str(train_root))
    num_classes = len(class_names)

    all_train_paths, all_train_labels = collect_image_paths(str(train_root), class_names)
    train_paths, train_labels, val_paths, val_labels = split_paths(all_train_paths, all_train_labels)
    test_paths, test_labels = collect_image_paths(str(test_root), class_names)

    train_ds = make_dataset_from_paths(train_paths, train_labels, batch_size=batch_size, shuffle=True)
    val_ds = make_dataset_from_paths(val_paths, val_labels, batch_size=batch_size, shuffle=False)
    test_ds = make_dataset_from_paths(test_paths, test_labels, batch_size=batch_size, shuffle=False)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    class_weights = build_class_weights(class_names, str(train_root))
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    _, test_accuracy = model.evaluate(test_ds, verbose=1)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)

    with open(LABELS_OUT, "w", encoding="utf-8") as handle:
        for label in class_names:
            handle.write(f"{label}\n")

    train_accuracy = history.history.get("accuracy", [None])[-1]
    validation_accuracy = history.history.get("val_accuracy", [None])[-1]
    print("Saved model to", MODEL_OUT)
    print("Saved labels to", LABELS_OUT)
    if train_accuracy is not None:
        print(f"Training accuracy: {train_accuracy:.4f}")
    if validation_accuracy is not None:
        print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    build_and_train()

"""Train a handwriting OCR CNN and export deployable artifacts.

The script expects an augmented dataset prepared by `scripts/prepare_dataset.py`.
That directory must contain `class_labels.json` and three subfolders:
`train/`, `val/`, and `test/`, each mirroring the original category structure and
holding 128×128 grayscale PNG images. Filenames follow the pattern
`<safe_label>__*.png`, and the class-label map resolves each `safe_label` to the
original handwriting string (e.g., "prn" → "PRN").

Outputs:
* SavedModel directory (for debugging / re-export)
* Best-performing Keras model (`best_model.keras`)
* TensorFlow Lite model (`ocr_model.tflite`)
* Labels file (`labels.txt`) ordered to match the exported model
* Per-epoch CSV log (`training_log.csv`), JSON metrics, and loss/accuracy plots

Example usage:
    python scripts/train_ocr.py --dataset-dir Handwriting_Dataset_Augmented --output-dir models/ocr
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from scripts.dataset_utils import (
    IMAGE_EXTENSIONS,
    extract_safe_label_from_stem,
    load_class_label_map,
    sanitize_label,
)

AUTOTUNE = tf.data.AUTOTUNE


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class DatasetSplit:
    name: str
    samples: List[Tuple[Path, str]]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)


def limit_split_samples(split: DatasetSplit, limit: Optional[int], seed: int) -> DatasetSplit:
    if limit is None or limit >= len(split.samples) or limit <= 0:
        return split
    rng = random.Random(seed)
    sampled = rng.sample(split.samples, limit)
    return DatasetSplit(split.name, sampled)


@dataclass
class TrainingArtifacts:
    saved_model_dir: Path
    best_model_path: Path
    tflite_path: Path
    labels_path: Path
    history_path: Path
    plot_path: Path
    metrics_path: Path
    log_path: Path


def collect_split_samples(dataset_dir: Path, split_name: str, label_map: Dict[str, str]) -> DatasetSplit:
    split_dir = dataset_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory '{split_name}' not found under {dataset_dir}")

    samples: List[Tuple[Path, str]] = []
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        safe_label = extract_safe_label_from_stem(path.stem)
        label = label_map.get(safe_label, sanitize_label(path.stem.split("__")[0]))
        samples.append((path.resolve(), label))

    return DatasetSplit(split_name, samples)


def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def build_tf_dataset(
    split: DatasetSplit,
    label_encoder: LabelEncoder,
    image_size: int,
    batch_size: int,
    cache_dataset: bool,
    training: bool,
    seed: int,
) -> Optional[tf.data.Dataset]:
    if not split.samples:
        return None
    paths = [str(path) for path, _ in split.samples]
    labels = [label for _, label in split.samples]
    encoded_labels = label_encoder.transform(labels)

    ds = tf.data.Dataset.from_tensor_slices((paths, encoded_labels))
    if training:
        ds = ds.shuffle(buffer_size=max(len(paths), 1024), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: load_and_preprocess_image(p, l, image_size), num_parallel_calls=AUTOTUNE)
    if cache_dataset:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_cnn_model(image_size: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 1), name="input_image")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype=tf.float32)(x)
    return tf.keras.Model(inputs, outputs, name="handwriting_cnn")


def summarize_split(split: DatasetSplit) -> None:
    print(f"{split.name.capitalize()} split: {len(split.samples)} images")
    labels = [label for _, label in split.samples]
    unique = len(set(labels))
    print(f"  Classes: {unique}")
    if labels:
        top_counts = Counter(labels).most_common(3)
        top_str = ", ".join(f"{label} ({count})" for label, count in top_counts)
        print(f"  Top labels: {top_str}")


def plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    history_dict = history.history
    plt.figure(figsize=(10, 4))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history_dict.get("accuracy", []), label="Train acc")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="Val acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history_dict.get("loss", []), label="Train loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Val loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_tflite(model: tf.keras.Model, output_path: Path, quantize: bool = True) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def save_labels(labels: Sequence[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")


def train(args: argparse.Namespace) -> TrainingArtifacts:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_dir} ...")
    label_map = load_class_label_map(dataset_dir)
    train_split = collect_split_samples(dataset_dir, "train", label_map)
    val_split = collect_split_samples(dataset_dir, "val", label_map)
    test_split = collect_split_samples(dataset_dir, "test", label_map)

    train_split = limit_split_samples(train_split, args.train_sample_limit, args.seed)
    val_split = limit_split_samples(val_split, args.val_sample_limit, args.seed + 1)
    test_split = limit_split_samples(test_split, args.test_sample_limit, args.seed + 2)

    for split in (train_split, val_split, test_split):
        summarize_split(split)

    discovered_labels = {label for _, label in train_split.samples + val_split.samples + test_split.samples}
    if not discovered_labels:
        raise RuntimeError("No labels found in provided dataset splits.")
    label_priority: Dict[str, int] = {}
    for idx, (_safe_label, original_label) in enumerate(label_map.items()):
        label_priority.setdefault(original_label, idx)
    unique_labels = sorted(discovered_labels, key=lambda lbl: label_priority.get(lbl, sys.maxsize))
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    train_ds = build_tf_dataset(
        train_split,
        label_encoder,
        args.image_size,
        args.batch_size,
        args.cache,
        training=True,
        seed=args.seed,
    )
    val_ds = build_tf_dataset(
        val_split,
        label_encoder,
        args.image_size,
        args.batch_size,
        args.cache,
        training=False,
        seed=args.seed,
    )
    test_ds = build_tf_dataset(
        test_split,
        label_encoder,
        args.image_size,
        args.batch_size,
        args.cache,
        training=False,
        seed=args.seed,
    )

    if train_ds is None:
        raise RuntimeError("Training split is empty; cannot proceed.")

    model = build_cnn_model(args.image_size, len(label_encoder.classes_))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    best_model_path = output_dir / "best_model.keras"
    csv_log_path = output_dir / "training_log.csv"

    has_val_split = val_ds is not None
    monitor_metric = "val_accuracy" if has_val_split else "accuracy"
    reduce_monitor = "val_loss" if has_val_split else "loss"

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=monitor_metric,
            mode="max" if monitor_metric.endswith("accuracy") else "min",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=reduce_monitor,
            factor=0.5,
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if has_val_split and args.early_stopping_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
                mode="max",
            )
        )

    print("Starting training ...")
    training_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration = time.time() - training_start

    plot_path = output_dir / "training_curves.png"
    plot_history(history, plot_path)

    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    if best_model_path.exists():
        print("Loading best checkpoint weights ...")
        model = tf.keras.models.load_model(best_model_path)

    metrics_summary: Dict[str, float] = {
        "num_classes": int(len(label_encoder.classes_)),
        "train_samples": int(len(train_split.samples)),
        "val_samples": int(len(val_split.samples)),
        "test_samples": int(len(test_split.samples)),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "initial_lr": float(args.learning_rate),
        "epochs_trained": int(len(history.history.get("loss", []))),
        "reduce_lr_patience": int(args.reduce_lr_patience),
        "early_stopping_patience": int(args.early_stopping_patience),
        "training_seconds": float(training_duration),
    }
    if "accuracy" in history.history:
        metrics_summary["train_accuracy"] = float(history.history["accuracy"][-1])
    if "loss" in history.history:
        metrics_summary["train_loss"] = float(history.history["loss"][-1])
    print("Evaluating model ...")
    if has_val_split:
        val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
        metrics_summary.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        print(f"Validation accuracy: {val_metrics.get('accuracy', 0.0):.4f}")

    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
        metrics_summary.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
        print(f"Test accuracy: {test_metrics.get('accuracy', 0.0):.4f}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    saved_model_dir = output_dir / "saved_model"
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    model.export(saved_model_dir)

    tflite_path = output_dir / "ocr_model.tflite"
    print("Exporting TensorFlow Lite model ...")
    export_tflite(model, tflite_path, quantize=not args.disable_quantization)

    labels_path = output_dir / "labels.txt"
    save_labels(label_encoder.classes_, labels_path)

    return TrainingArtifacts(
        saved_model_dir=saved_model_dir,
        best_model_path=best_model_path,
        tflite_path=tflite_path,
        labels_path=labels_path,
        history_path=history_path,
        plot_path=plot_path,
        metrics_path=metrics_path,
        log_path=csv_log_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train handwriting OCR model and export TFLite bundle.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the prepared dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where training outputs will be stored")
    parser.add_argument("--image-size", type=int, default=128, help="Input image size (pixels)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--cache", action="store_true", help="Cache datasets in memory for faster training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Set to 0 to disable EarlyStopping (only used when validation split exists)",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement before reducing learning rate",
    )
    parser.add_argument(
        "--train-sample-limit",
        type=int,
        default=None,
        help="Sample at most this many training images (useful for smoke tests)",
    )
    parser.add_argument(
        "--val-sample-limit",
        type=int,
        default=None,
        help="Sample at most this many validation images",
    )
    parser.add_argument(
        "--test-sample-limit",
        type=int,
        default=None,
        help="Sample at most this many test images",
    )
    parser.add_argument(
        "--disable-quantization",
        action="store_true",
        help="Disable post-training dynamic range quantization when exporting TFLite",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Set TensorFlow intra/inter op parallelism threads (improves reproducibility)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable GPU usage even if one is available",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable float16 mixed-precision training when a GPU is available",
    )

    args = parser.parse_args(argv)
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be >= 0")
    if args.reduce_lr_patience <= 0:
        parser.error("--reduce-lr-patience must be > 0")
    for flag_name in ("train_sample_limit", "val_sample_limit", "test_sample_limit"):
        value = getattr(args, flag_name)
        if value is not None and value <= 0:
            parser.error(f"--{flag_name.replace('_', '-')} must be positive when provided")
    if args.num_threads is not None and args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    return args


def configure_tensorflow_threads(num_threads: int | None) -> None:
    if num_threads is None:
        return
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)


def configure_accelerator(force_cpu: bool, enable_mixed_precision: bool) -> str:
    if force_cpu:
        tf.config.set_visible_devices([], "GPU")
        print("GPU usage disabled by flag; running on CPU.")
        if enable_mixed_precision:
            print("(note) Mixed precision requires GPU support; disabling.")
        return "CPU"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Detected {len(gpus)} GPU(s); TensorFlow will utilize them.")
            if enable_mixed_precision:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("Enabled mixed precision (float16) for GPU acceleration.")
            return "GPU"
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"Warning: GPU configuration failed ({exc}); falling back to CPU.")
            tf.config.set_visible_devices([], "GPU")
            if enable_mixed_precision:
                print("(note) Mixed precision disabled due to CPU fallback.")
            return "CPU"

    print("No compatible GPU detected; training will run on CPU.")
    if enable_mixed_precision:
        print("(note) Mixed precision requires GPU support; disabling.")
    return "CPU"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_tensorflow_threads(args.num_threads)
    configure_accelerator(args.force_cpu, args.mixed_precision)
    set_global_seed(args.seed)
    start_time = time.time()
    artifacts = train(args)
    elapsed = time.time() - start_time
    print("Training complete in {:.1f} minutes".format(elapsed / 60.0))
    print("Artifacts saved to:")
    print(f"  SavedModel: {artifacts.saved_model_dir}")
    print(f"  Best Keras model: {artifacts.best_model_path}")
    print(f"  TFLite model: {artifacts.tflite_path}")
    print(f"  Labels: {artifacts.labels_path}")
    print(f"  History: {artifacts.history_path}")
    print(f"  Metrics: {artifacts.metrics_path}")
    print(f"  Training curves: {artifacts.plot_path}")
    print(f"  Training log: {artifacts.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

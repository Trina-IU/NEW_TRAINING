"""Train a handwriting OCR classification model and export a TensorFlow Lite bundle.

This script expects the dataset that lives in `Handwriting_Dataset/` where each
image filename corresponds to the ground-truth label (e.g. `Paracetamol.jpg`,
`Take 1 tsp every 6 hours (2).png`). The parent folders are ignored for the
labels; only the file stem (minus optional suffixes like ` (2)`) is used.

Outputs:
* SavedModel directory (for debugging / re-export)
* Quantized TensorFlow Lite model (`ocr_model.tflite`)
* Plain-text labels file (`labels.txt`) ordered to match the model outputs
* Training history plot and JSON metrics summary

Example usage (from the project root):
    python scripts/train_ocr.py --dataset-dir Handwriting_Dataset --output-dir models/ocr
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import mixed_precision

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetSplit:
    name: str
    samples: List[Tuple[Path, str]]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)


@dataclass
class TrainingArtifacts:
    saved_model_dir: Path
    tflite_path: Path
    labels_path: Path
    history_path: Path
    plot_path: Path
    metrics_path: Path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def sanitize_label(stem: str) -> str:
    """Clean up a filename stem into a canonical label string."""
    normalized = unicodedata.normalize("NFKC", stem)
    # Remove trailing variants like " (2)" or " (12)"
    normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
    # Collapse underscores and multiple spaces into single spaces
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    if not normalized:
        raise ValueError(f"Empty label derived from stem '{stem}'")
    return normalized


def find_image_files(dataset_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = sanitize_label(path.stem)
        samples.append((path.resolve(), label))
    if not samples:
        raise FileNotFoundError(
            f"No image files with extensions {sorted(IMAGE_EXTENSIONS)} were found under {dataset_dir}."
        )
    samples.sort(key=lambda item: str(item[0]))
    return samples


def maybe_limit_samples(samples: List[Tuple[Path, str]], max_samples: int | None, seed: int) -> List[Tuple[Path, str]]:
    if max_samples is None or max_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    selected = indices[:max_samples]
    return [samples[i] for i in selected]


def split_samples(
    samples: Sequence[Tuple[Path, str]],
    val_split: float,
    test_split: float,
    seed: int,
) -> Dict[str, DatasetSplit]:
    by_label: Dict[str, List[Path]] = defaultdict(list)
    for path, label in samples:
        by_label[label].append(path)

    rng = random.Random(seed)
    train_samples: List[Tuple[Path, str]] = []
    val_samples: List[Tuple[Path, str]] = []
    test_samples: List[Tuple[Path, str]] = []

    for label, paths in by_label.items():
        label_rng = random.Random(seed + hash(label) % (2**31))
        shuffled_paths = list(paths)
        label_rng.shuffle(shuffled_paths)
        n = len(shuffled_paths)
        n_test = int(round(n * test_split)) if test_split > 0 else 0
        n_val = int(round(n * val_split)) if val_split > 0 else 0

        if val_split > 0 and n >= 2 and n_val == 0:
            n_val = 1
        if test_split > 0 and (n - n_val) >= 2 and n_test == 0:
            n_test = 1

        # Ensure we always retain at least one training sample per label
        while n_test + n_val >= n and n_test > 0:
            n_test -= 1
        while n_test + n_val >= n and n_val > 0:
            n_val -= 1
        if n - n_test - n_val <= 0:
            # As a final fallback, send everything to train
            n_test = 0
            n_val = 0

        train_end = n - (n_val + n_test)
        val_end = train_end + n_val

        train_samples.extend((path, label) for path in shuffled_paths[:train_end])
        val_samples.extend((path, label) for path in shuffled_paths[train_end:val_end])
        test_samples.extend((path, label) for path in shuffled_paths[val_end:])

    # Shuffle each split for better variety when batching
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return {
        "train": DatasetSplit("train", train_samples),
        "val": DatasetSplit("val", val_samples),
        "test": DatasetSplit("test", test_samples),
    }


def describe_dataset(samples: Sequence[Tuple[Path, str]]) -> Counter:
    labels = [label for _, label in samples]
    counts = Counter(labels)
    return counts


def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def build_tf_dataset(
    samples: DatasetSplit,
    label_encoder: LabelEncoder,
    image_size: int,
    batch_size: int,
    shuffle_buffer: int,
    cache_dataset: bool,
    training: bool,
) -> tf.data.Dataset:
    if len(samples) == 0:
        return None  # type: ignore

    paths = [str(path) for path, _ in samples.samples]
    labels = [label for _, label in samples.samples]
    encoded_labels = label_encoder.transform(labels)

    ds = tf.data.Dataset.from_tensor_slices((paths, encoded_labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), shuffle_buffer), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: load_and_preprocess_image(p, l, image_size), num_parallel_calls=AUTOTUNE)
    if cache_dataset:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_model(image_size: int, num_classes: int, base_trainable: bool) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = base_trainable

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="handwriting_ocr")
    return model


def enable_fine_tuning(model: tf.keras.Model, unfreeze_fraction: float) -> None:
    if not 0.0 < unfreeze_fraction <= 1.0:
        raise ValueError("unfreeze_fraction must be within (0, 1]")
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
            base_model = layer
            break
    if base_model is None:
        raise RuntimeError("Could not locate base MobileNetV2 model for fine-tuning")
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1.0 - unfreeze_fraction))
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    print(f"Fine-tuning enabled: unfroze {total_layers - unfreeze_from} layers out of {total_layers}.")


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
    samples = find_image_files(dataset_dir)
    samples = maybe_limit_samples(samples, args.max_samples, args.seed)
    print(f"Discovered {len(samples)} images across {len(set(label for _, label in samples))} labels.")

    counts = describe_dataset(samples)
    print("Dataset class distribution (top 10):")
    for label, count in counts.most_common(10):
        print(f"  {label}: {count}")
    if len(counts) > 10:
        print("  ...")

    splits = split_samples(samples, args.val_split, args.test_split, args.seed)
    for split_name, split in splits.items():
        print(f"{split_name.capitalize()} split: {len(split)} samples")
        if len(split) == 0:
            print(f"  (warning) No samples in {split_name} split")

    label_encoder = LabelEncoder()
    label_encoder.fit([label for _, label in samples])

    train_ds = build_tf_dataset(
        splits["train"],
        label_encoder,
        args.image_size,
        args.batch_size,
        args.shuffle_buffer,
        args.cache,
        training=True,
    )
    val_ds = build_tf_dataset(
        splits["val"],
        label_encoder,
        args.image_size,
        args.batch_size,
        args.shuffle_buffer,
        args.cache,
        training=False,
    )
    test_ds = build_tf_dataset(
        splits["test"],
        label_encoder,
        args.image_size,
        args.batch_size,
        args.shuffle_buffer,
        args.cache,
        training=False,
    )

    steps_per_epoch = None
    if args.max_steps_per_epoch:
        steps_per_epoch = args.max_steps_per_epoch

    model = build_model(args.image_size, len(label_encoder.classes_), base_trainable=args.fine_tune_fraction > 0)

    if args.fine_tune_fraction > 0:
        enable_fine_tuning(model, args.fine_tune_fraction)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks: List[tf.keras.callbacks.Callback] = []
    if args.early_stopping_patience > 0 and val_ds is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
                mode="max",
            )
        )
    checkpoint_path = output_dir / "best_weights.keras"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy" if val_ds is not None else "accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        )
    )

    print("Starting training ...")
    if steps_per_epoch is not None:
        train_ds = train_ds.repeat()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    plot_path = output_dir / "training_curves.png"
    plot_history(history, plot_path)

    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    if checkpoint_path.exists():
        print("Loading best checkpoint weights ...")
        model = tf.keras.models.load_model(checkpoint_path)

    metrics_summary: Dict[str, float] = {}
    print("Evaluating model ...")
    if val_ds is not None:
        val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
        metrics_summary.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        print(f"Validation accuracy: {val_metrics.get('accuracy', 'n/a'):.4f}")
    if test_ds is not None and len(splits["test"]) > 0:
        test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
        metrics_summary.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
        print(f"Test accuracy: {test_metrics.get('accuracy', 'n/a'):.4f}")

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
        tflite_path=tflite_path,
        labels_path=labels_path,
        history_path=history_path,
        plot_path=plot_path,
        metrics_path=metrics_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train handwriting OCR model and export TFLite bundle.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the handwriting dataset root directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where training outputs will be stored")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (pixels)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of samples reserved for validation")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction of samples reserved for hold-out testing")
    parser.add_argument("--shuffle-buffer", type=int, default=512, help="Buffer size for dataset shuffling")
    parser.add_argument("--cache", action="store_true", help="Cache datasets in memory for faster training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on number of samples (for debugging)")
    parser.add_argument(
        "--fine-tune-fraction",
        type=float,
        default=0.0,
        help="Fraction of backbone layers to unfreeze for fine-tuning (0 disables fine-tuning)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Epoch patience for early stopping when validation accuracy plateaus",
    )
    parser.add_argument(
        "--disable-quantization",
        action="store_true",
        help="Disable post-training dynamic range quantization when exporting TFLite",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Optional maximum number of steps per epoch (useful for smoke tests)",
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
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive when provided")
    if args.val_split < 0 or args.test_split < 0 or (args.val_split + args.test_split) >= 1:
        parser.error("val_split and test_split must be >=0 and sum to < 1")
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
                mixed_precision.set_global_policy("mixed_float16")
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
    print(f"  TFLite model: {artifacts.tflite_path}")
    print(f"  Labels: {artifacts.labels_path}")
    print(f"  History: {artifacts.history_path}")
    print(f"  Metrics: {artifacts.metrics_path}")
    print(f"  Training curves: {artifacts.plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

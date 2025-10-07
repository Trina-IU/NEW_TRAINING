"""Compute confusion matrix and per-class stats using the trained Keras model.

This script loads a Keras model, the labels file, and runs predictions on a
chosen split of the prepared dataset. It saves a CSV confusion matrix and a
per-class summary (support, correct, accuracy). Helpful to find dominant
classes or systematic confusions.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
import tensorflow as tf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.dataset_utils import IMAGE_EXTENSIONS, extract_safe_label_from_stem, load_class_label_map


def load_labels(path: Path) -> List[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def collect_images(split_dir: Path) -> List[Path]:
    return [p for p in sorted(split_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]


def load_image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description="Compute confusion matrix for Keras model on dataset split")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--keras-model", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-matrix", default="reports/confusion_matrix.csv")
    p.add_argument("--out-summary", default="reports/confusion_summary.csv")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.keras_model)
    labels = load_labels(Path(args.labels))
    label_to_idx: Dict[str, int] = {l: i for i, l in enumerate(labels)}

    split_dir = dataset_dir / args.split
    images = collect_images(split_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images found in {split_dir}")
        return 2

    model = tf.keras.models.load_model(str(model_path))

    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    support = defaultdict(int)
    correct = defaultdict(int)

    for idx, p in enumerate(images, start=1):
        safe = extract_safe_label_from_stem(p.stem)
        true_label = load_class_label_map(dataset_dir).get(safe)
        if true_label is None:
            print(f"Warning: no label for {p}; skipping")
            continue
        if true_label not in label_to_idx:
            print(f"Warning: true label '{true_label}' not in labels file; skipping")
            continue
        y_true = label_to_idx[true_label]
        arr = load_image(p, args.image_size)
        inp = arr.reshape((1, args.image_size, args.image_size, 1)).astype(np.float32)
        probs = model.predict(inp, verbose=0).ravel()
        y_pred = int(np.argmax(probs))
        cm[y_true, y_pred] += 1
        support[true_label] += 1
        if y_true == y_pred:
            correct[true_label] += 1

        if idx % 100 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)}", end='\r')

    # save confusion matrix (rows=true, cols=predicted)
    out_dir = Path(args.out_matrix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_matrix, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, lab in enumerate(labels):
            writer.writerow([lab] + cm[i].tolist())

    # summary per class
    with open(args.out_summary, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "support", "correct", "accuracy"])
        for lab in labels:
            s = support.get(lab, 0)
            c = correct.get(lab, 0)
            acc = (c / s) if s > 0 else 0.0
            writer.writerow([lab, s, c, f"{acc:.4f}"])

    print("Confusion matrix saved to:", args.out_matrix)
    print("Per-class summary saved to:", args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

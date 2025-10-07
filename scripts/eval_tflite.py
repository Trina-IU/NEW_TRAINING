"""Evaluate a trained TFLite handwriting OCR model on a prepared dataset split.

This script reads the augmented dataset produced by `scripts/prepare_dataset.py`,
loads the exported `ocr_model.tflite`, and computes accuracy metrics on the
chosen split (train/val/test). It can also emit a CSV containing any
misclassified samples for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

if __package__ is None or __package__ == "":  # pragma: no cover - direct invocation helper
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.dataset_utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    extract_safe_label_from_stem,
    load_class_label_map,
)


@dataclass
class SampleResult:
    path: Path
    true_label: str
    predicted_label: str
    topk_labels: List[Tuple[str, float]]
    score: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR TFLite model on prepared dataset split.")
    parser.add_argument("--dataset-dir", required=True, type=str, help="Prepared dataset root (with train/val/test)")
    parser.add_argument("--model-path", required=True, type=str, help="Path to ocr_model.tflite")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Optional labels.txt path (defaults to sibling of model)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of samples for quicker evaluation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Report accuracy within top-k predictions (default: 3)",
    )
    parser.add_argument(
        "--misclass-report",
        type=str,
        default=None,
        help="Optional CSV file to save misclassified samples",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Expected input image size (auto-detected when possible)",
    )
    args = parser.parse_args(argv)
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive when provided")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def load_labels(labels_path: Path) -> List[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"labels file {labels_path} is empty")
    return labels


def collect_image_paths(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    images = [path for path in sorted(split_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise RuntimeError(f"No images found under {split_dir}")
    return images


def load_image(path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
    return arr


def prepare_input_tensor(arr: np.ndarray, input_details: Dict[str, np.ndarray]) -> np.ndarray:
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], 1))
    dtype = input_details["dtype"]
    if dtype == np.float32:
        arr = arr / 255.0
        return arr.astype(np.float32)
    if dtype in (np.uint8, np.int8):
        scale = input_details.get("quantization_parameters", {}).get("scales")
        zero_point = input_details.get("quantization_parameters", {}).get("zero_points")
        if scale is None or len(scale) == 0:
            raise ValueError("Quantized model missing scale information")
        scale_value = float(scale[0])
        zero_point_value = int(zero_point[0]) if zero_point is not None else 0
        quantized = arr / 255.0 / scale_value + zero_point_value
        quantized = np.clip(np.round(quantized), np.iinfo(dtype).min, np.iinfo(dtype).max)
        return quantized.astype(dtype)
    raise NotImplementedError(f"Unsupported TFLite input dtype: {dtype}")


def interpret_output(output: np.ndarray, labels: List[str]) -> Tuple[str, float, List[Tuple[str, float]]]:
    probs = output.astype(np.float32).ravel()
    probs = probs / np.sum(probs)
    top_indices = probs.argsort()[::-1]
    best_idx = int(top_indices[0])
    best_label = labels[best_idx]
    best_score = float(probs[best_idx])
    topk = [(labels[int(idx)], float(probs[int(idx)])) for idx in top_indices]
    return best_label, best_score, topk


def evaluate_split(
    dataset_dir: Path,
    model_path: Path,
    labels_path: Path,
    split: str,
    sample_limit: Optional[int],
    top_k: int,
    image_size: int,
    report_path: Optional[Path],
) -> None:
    label_map = load_class_label_map(dataset_dir)
    labels = load_labels(labels_path)
    label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}

    split_dir = dataset_dir / split
    image_paths = collect_image_paths(split_dir)
    if sample_limit is not None:
        image_paths = image_paths[:sample_limit]

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_height = int(input_details["shape"][1]) if input_details["shape"][1] > 0 else image_size
    input_width = int(input_details["shape"][2]) if input_details["shape"][2] > 0 else image_size
    expected_size = (input_width, input_height)

    results: List[SampleResult] = []
    misclassified: List[SampleResult] = []
    inference_times: List[float] = []

    total = len(image_paths)
    topk_hits = 0
    correct = 0

    for idx, path in enumerate(image_paths, start=1):
        safe_label = extract_safe_label_from_stem(path.stem)
        true_label = label_map.get(safe_label)
        if true_label is None:
            raise KeyError(f"Label for safe slug '{safe_label}' not found in class_labels.json")
        if true_label not in label_to_index:
            raise KeyError(f"True label '{true_label}' not found in labels file {labels_path}")

        arr = load_image(path, expected_size)
        input_tensor = prepare_input_tensor(arr, input_details)

        interpreter.set_tensor(input_details["index"], input_tensor)
        start = time.perf_counter()
        interpreter.invoke()
        inference_times.append(time.perf_counter() - start)
        output = interpreter.get_tensor(output_details["index"])

        predicted_label, score, topk = interpret_output(output, labels)
        result = SampleResult(path=path, true_label=true_label, predicted_label=predicted_label, topk_labels=topk[:top_k], score=score)
        results.append(result)

        topk_labels_only = {label for label, _ in topk[:top_k]}
        if predicted_label == true_label:
            correct += 1
        else:
            misclassified.append(result)
        if true_label in topk_labels_only:
            topk_hits += 1

        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} samples...", end="\r")

    print()
    accuracy = correct / total
    topk_accuracy = topk_hits / total
    avg_latency_ms = statistics.mean(inference_times) * 1000.0
    p95_latency_ms = statistics.quantiles(inference_times, n=20)[-1] * 1000.0 if len(inference_times) >= 20 else max(inference_times) * 1000.0

    print("Evaluation summary:")
    print(f"  Split:          {split}")
    print(f"  Samples:        {total}")
    print(f"  Top-1 accuracy: {accuracy:.4f}")
    print(f"  Top-{top_k} accuracy: {topk_accuracy:.4f}")
    print(f"  Avg latency:    {avg_latency_ms:.2f} ms")
    print(f"  P95 latency:    {p95_latency_ms:.2f} ms")
    print(f"  Misclassified:  {len(misclassified)}")

    if report_path and misclassified:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "true_label", "predicted_label", "confidence", "topk_predictions"])
            for item in misclassified:
                rel_path = item.path.relative_to(dataset_dir).as_posix()
                topk_str = "; ".join(f"{label}:{score:.3f}" for label, score in item.topk_labels)
                writer.writerow([rel_path, item.true_label, item.predicted_label, f"{item.score:.4f}", topk_str])
        print(f"Misclassified sample report saved to {report_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    labels_path = Path(args.labels_path).expanduser().resolve() if args.labels_path else model_path.with_name("labels.txt")
    report_path = Path(args.misclass_report).expanduser().resolve() if args.misclass_report else None

    evaluate_split(
        dataset_dir=dataset_dir,
        model_path=model_path,
        labels_path=labels_path,
        split=args.split,
        sample_limit=args.sample_limit,
        top_k=args.top_k,
        image_size=args.image_size,
        report_path=report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

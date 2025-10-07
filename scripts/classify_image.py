"""Classify one or more handwriting images with the exported TFLite model.

Usage example:

    python scripts/classify_image.py \
        --model models/ocr/ocr_model.tflite \
        --labels models/ocr/labels.txt \
        --images sample1.png sample2.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

if __package__ is None or __package__ == "":  # allow `python scripts/classify_image.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.dataset_utils import IMAGE_EXTENSIONS  # noqa: E402


def load_labels(path: Path) -> List[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Labels file {path} is empty")
    return labels


def load_image(path: Path, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
    return arr


def prepare_tensor(arr: np.ndarray, input_details: dict) -> np.ndarray:
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], 1))
    dtype = input_details["dtype"]
    if dtype == np.float32:
        return (arr / 255.0).astype(np.float32)
    if dtype in (np.uint8, np.int8):
        quant = input_details.get("quantization_parameters") or {}
        scales = quant.get("scales")
        zero_points = quant.get("zero_points")
        if scales is None or len(scales) == 0:
            raise ValueError("Quantized model missing scale information")
        scale = float(scales[0])
        zero_point = int(zero_points[0]) if zero_points is not None else 0
        quantized = arr / 255.0 / scale + zero_point
        qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
        return np.clip(np.round(quantized), qmin, qmax).astype(dtype)
    raise NotImplementedError(f"Unsupported TFLite input dtype: {dtype}")


def topk(probs: np.ndarray, labels: List[str], k: int = 3) -> List[Tuple[str, float]]:
    probs = probs / probs.sum()
    indices = probs.argsort()[::-1][:k]
    return [(labels[int(i)], float(probs[int(i)])) for i in indices]


def iter_image_paths(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            out.extend(sorted(child for child in p.rglob("*") if child.suffix.lower() in IMAGE_EXTENSIONS))
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError("No images found for given paths")
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify handwriting images with a TFLite OCR model")
    parser.add_argument("--model", required=True, type=str, help="Path to ocr_model.tflite")
    parser.add_argument("--labels", required=True, type=str, help="Path to labels.txt")
    parser.add_argument("--images", nargs="+", help="Image files or directories containing images")
    parser.add_argument("--top-k", type=int, default=3, help="Report top-k predictions (default 3)")
    parser.add_argument("--image-size", type=int, default=128, help="Resize images to this square size")
    parser.add_argument("--quiet", action="store_true", help="Suppress interpreter warnings")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    labels = load_labels(Path(args.labels))
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_h = int(input_details["shape"][1]) if input_details["shape"][1] > 0 else args.image_size
    input_w = int(input_details["shape"][2]) if input_details["shape"][2] > 0 else args.image_size
    target_size = (input_w, input_h)

    image_paths = iter_image_paths(args.images)
    for path in image_paths:
        arr = load_image(path, target_size)
        tensor = prepare_tensor(arr, input_details)
        interpreter.set_tensor(input_details["index"], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"]).ravel().astype(np.float32)
        predictions = topk(output, labels, args.top_k)

        print(f"\nImage: {path}")
        for rank, (label, score) in enumerate(predictions, start=1):
            print(f"  {rank}. {label:25s} {score:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

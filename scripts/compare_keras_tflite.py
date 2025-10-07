"""Compare Keras (.keras) model predictions against a TFLite model.

Runs both models on the first N images from a chosen split and prints top-3
predictions side-by-side so you can spot label-ordering or quantization issues.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.dataset_utils import IMAGE_EXTENSIONS, extract_safe_label_from_stem, load_class_label_map


def load_labels(labels_path: Path) -> List[str]:
    return [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_image(path: Path, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def topk_from_probs(probs: np.ndarray, labels: List[str], k: int = 3):
    idx = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in idx]


def main():
    import argparse

    p = argparse.ArgumentParser(description="Compare Keras and TFLite predictions")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--keras-model", required=True)
    p.add_argument("--tflite-model", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    keras_path = Path(args.keras_model)
    tflite_path = Path(args.tflite_model)
    labels_path = Path(args.labels)

    labels = load_labels(labels_path)
    label_map = load_class_label_map(dataset_dir)

    # collect images
    split_dir = dataset_dir / args.split
    images = [p for p in sorted(split_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    images = images[: args.limit]
    if not images:
        print("No images found in split", split_dir)
        return

    # load keras model
    keras = tf.keras.models.load_model(str(keras_path))
    # tflite interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_h = int(in_det['shape'][1]) if in_det['shape'][1] > 0 else 128
    in_w = int(in_det['shape'][2]) if in_det['shape'][2] > 0 else 128

    print(f"Comparing {len(images)} images; model input size: {in_w}x{in_h}; labels: {len(labels)}")

    for img_path in images:
        arr = load_image(img_path, (in_w, in_h))
        inp = arr.reshape((1, in_h, in_w, 1)).astype(np.float32)

        # keras prediction
        k_probs = keras.predict(inp, verbose=0).ravel()
        k_top = topk_from_probs(k_probs, labels, 3)

        # tflite prediction
        dtype = in_det['dtype']
        if dtype == np.float32:
            t_input = inp.astype(np.float32)
        else:
            # handle simple quantized cases
            scale, zero = (in_det.get('quantization', (0.0, 0))[0], in_det.get('quantization', (0, 0))[1])
            if scale == 0:
                t_input = (inp * 255.0).astype(dtype)
            else:
                t_input = np.clip(np.round(inp / scale + zero), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

        interpreter.set_tensor(in_det['index'], t_input)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index']).ravel().astype(np.float32)
        # if quantized, we can't easily recover scale/zero here without more details; assume softmax-like
        try:
            t_probs = out / out.sum()
        except Exception:
            t_probs = out
        t_top = topk_from_probs(t_probs, labels, 3)

        safe = extract_safe_label_from_stem(img_path.stem)
        true_label = label_map.get(safe, "<unknown>")

        print(f"\nImage: {img_path.relative_to(dataset_dir)}  true: {true_label}")
        print("  Keras top3:", k_top)
        print("  TFLite top3:", t_top)


if __name__ == '__main__':
    main()

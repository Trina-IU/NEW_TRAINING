"""Dataset preparation pipeline for handwriting OCR.

Steps performed:
1. Rename Windows-reserved filenames (e.g., PRN) to safe variants.
2. Drop unreadable or zero-byte images.
3. Convert all images to 128x128 grayscale, scaling pixel values to [0, 1].
4. Generate N augmented variants per source image using Keras augmentation.
5. Split into train/val/test (80/10/10) while preserving original subfolder structure.
6. Save a class label map for use during training.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from scripts.dataset_utils import (
    IMAGE_EXTENSIONS,
    is_windows_reserved,
    make_unique_slug,
    sanitize_label,
    save_class_label_map,
)


@dataclass
class ImageEntry:
    path: Path
    relative_parent: Path
    safe_label: str
    label: str


class LabelRegistry:
    def __init__(self) -> None:
        self.label_to_safe: Dict[str, str] = {}
        self.safe_to_label: Dict[str, str] = {}

    def register(self, label: str) -> str:
        if label in self.label_to_safe:
            return self.label_to_safe[label]
        safe = make_unique_slug(label, self.label_to_safe)
        safe = safe.lower()
        self.label_to_safe[label] = safe
        self.safe_to_label[safe] = label
        return safe


def rename_reserved_names(dataset_dir: Path) -> List[tuple[Path, Path]]:
    renames: List[tuple[Path, Path]] = []
    for root, dirnames, filenames in os.walk(dataset_dir):
        # Directories
        for idx, dirname in enumerate(dirnames):
            if is_windows_reserved(dirname):
                src = Path(root) / dirname
                suffix = 0
                while True:
                    candidate = f"_{dirname}" if suffix == 0 else f"_{dirname}_{suffix}"
                    dst = Path(root) / candidate
                    if not dst.exists():
                        break
                    suffix += 1
                src.rename(dst)
                dirnames[idx] = dst.name
                renames.append((src, dst))
        # Files
        for filename in filenames:
            if is_windows_reserved(filename):
                src = Path(root) / filename
                stem, ext = os.path.splitext(filename)
                suffix = 0
                while True:
                    candidate_stem = f"_{stem}" if suffix == 0 else f"_{stem}_{suffix}"
                    dst = Path(root) / f"{candidate_stem}{ext}"
                    if not dst.exists():
                        break
                    suffix += 1
                src.rename(dst)
                renames.append((src, dst))
    return renames


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def preprocess_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        img = ImageOps.fit(img, (image_size, image_size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def save_grayscale_image(arr: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr_uint8, mode="L").save(dest)


def collect_images(dataset_dir: Path) -> List[Path]:
    return sorted(path for path in dataset_dir.rglob("*") if is_image_file(path))


def remove_corrupted_and_zero_byte(images: Sequence[Path]) -> List[Path]:
    cleaned: List[Path] = []
    for path in images:
        try:
            if path.stat().st_size == 0:
                path.unlink(missing_ok=True)
                continue
        except FileNotFoundError:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            # reopen to ensure load works after verify
            with Image.open(path) as img:
                img.convert("RGB").load()
        except Exception:
            path.unlink(missing_ok=True)
            continue
        cleaned.append(path)
    return cleaned


def build_datagen(seed: int) -> ImageDataGenerator:
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        horizontal_flip=False,
        fill_mode="nearest",
    )


def generate_augmented_dataset(
    dataset_dir: Path,
    output_dir: Path,
    image_size: int,
    augmentations_per_image: int,
    seed: int,
) -> tuple[List[ImageEntry], Dict[str, str]]:
    stage_dir = output_dir / "_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    datagen = build_datagen(seed)
    registry = LabelRegistry()
    label_counts: Dict[str, int] = defaultdict(int)
    metadata: List[ImageEntry] = []

    images = collect_images(dataset_dir)
    images = remove_corrupted_and_zero_byte(images)
    if not images:
        raise RuntimeError("No valid images found after cleanup.")

    for path in images:
        label = sanitize_label(path.stem)
        safe_label = registry.register(label)
        relative_parent = path.relative_to(dataset_dir).parent

        arr = preprocess_image(path, image_size)
        base_index = label_counts[safe_label]
        base_filename = f"{safe_label}__orig_{base_index:04d}.png"
        base_dest = stage_dir / relative_parent / base_filename
        save_grayscale_image(arr, base_dest)
        metadata.append(ImageEntry(path=base_dest, relative_parent=relative_parent, safe_label=safe_label, label=label))
        label_counts[safe_label] += 1

        tensor = (arr * 255.0).reshape((1, image_size, image_size, 1))
        flow_seed = rng.randint(0, 10**9)
        flow = datagen.flow(tensor, batch_size=1, seed=flow_seed)
        for _ in range(augmentations_per_image):
            augmented = next(flow)[0] / 255.0
            augmented = np.clip(augmented, 0.0, 1.0)
            aug_index = label_counts[safe_label]
            aug_filename = f"{safe_label}__aug_{aug_index:04d}.png"
            aug_dest = stage_dir / relative_parent / aug_filename
            save_grayscale_image(augmented.squeeze(), aug_dest)
            metadata.append(ImageEntry(path=aug_dest, relative_parent=relative_parent, safe_label=safe_label, label=label))
            label_counts[safe_label] += 1

    return metadata, registry.safe_to_label


def split_entries(
    entries: Sequence[ImageEntry],
    output_dir: Path,
    seed: int,
) -> Dict[str, List[ImageEntry]]:
    grouped: Dict[str, List[ImageEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.safe_label].append(entry)

    rng = random.Random(seed)
    split_records: Dict[str, List[ImageEntry]] = {"train": [], "val": [], "test": []}

    for safe_label, group in grouped.items():
        group_copy = list(group)
        rng.shuffle(group_copy)
        n = len(group_copy)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        n_test = n - n_train - n_val

        if n_val == 0 and n >= 3:
            n_val = 1
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_val
        if n_test == 0 and n - n_train - n_val > 0:
            n_test = 1
            if n_train > n_val:
                n_train = max(1, n_train - 1)
            else:
                n_val = max(0, n_val - 1)

        if n_train + n_val + n_test != n:
            diff = n - (n_train + n_val + n_test)
            n_train += diff

        idx = 0
        split_records["train"].extend(group_copy[idx : idx + n_train])
        idx += n_train
        split_records["val"].extend(group_copy[idx : idx + n_val])
        idx += n_val
        split_records["test"].extend(group_copy[idx:])

    # Move files into final directories
    manifest: Dict[str, List[tuple[str, str, str]]] = {"train": [], "val": [], "test": []}
    for split_name, split_entries_list in split_records.items():
        for entry in split_entries_list:
            dest_dir = output_dir / split_name / entry.relative_parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / entry.path.name
            shutil.move(entry.path, dest_path)
            manifest[split_name].append(
                (
                    dest_path.relative_to(output_dir).as_posix(),
                    entry.safe_label,
                    entry.label,
                )
            )

    stage_dir = output_dir / "_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    for split_name, rows in manifest.items():
        manifest_path = output_dir / f"{split_name}_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "safe_label", "label"])
            writer.writerows(sorted(rows))

    return split_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare handwriting dataset with augmentation and splitting.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the original dataset root")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where the augmented dataset will be saved")
    parser.add_argument("--image-size", type=int, default=128, help="Target square image size (pixels)")
    parser.add_argument("--augmentations-per-image", type=int, default=10, help="Number of augmented variants per original image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it already exists")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory {output_dir} already exists. Use --force to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Renaming Windows-reserved filenames in {dataset_dir} (if any)...")
    renames = rename_reserved_names(dataset_dir)
    if renames:
        for src, dst in renames:
            print(f"  {src.name} -> {dst.name}")
    else:
        print("  No reserved filenames detected.")

    print("Generating normalized and augmented dataset...")
    metadata, safe_to_label = generate_augmented_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        image_size=args.image_size,
        augmentations_per_image=args.augmentations_per_image,
        seed=args.seed,
    )
    print(f"  Prepared {len(metadata)} processed images across {len(safe_to_label)} labels.")

    print("Splitting into train/val/test (80/10/10)...")
    splits = split_entries(metadata, output_dir=output_dir, seed=args.seed)
    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)} images")

    mapping_path = output_dir / "class_labels.json"
    save_class_label_map(safe_to_label, mapping_path)
    print(f"Saved class label map to {mapping_path}")

    print("Dataset preparation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

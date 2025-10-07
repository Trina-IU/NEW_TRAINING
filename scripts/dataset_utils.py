"""Shared dataset utilities for OCR training pipeline."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Windows reserved device names (case-insensitive)
RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def sanitize_label(stem: str) -> str:
    """Normalize a filename stem into a human-readable label."""
    normalized = unicodedata.normalize("NFKC", stem)
    normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError(f"Empty label derived from stem '{stem}'")
    return normalized


_slug_invalid_pattern = re.compile(r"[^0-9A-Za-z]+")


def slugify_label(label: str) -> str:
    """Generate a filesystem-friendly slug from a label."""
    normalized = unicodedata.normalize("NFKC", label)
    slug = _slug_invalid_pattern.sub("_", normalized).strip("_").lower()
    if not slug:
        slug = "label"
    return slug


def make_unique_slug(label: str, existing: Dict[str, str]) -> str:
    """Return a unique slug for the given label, tracking existing assignments."""
    base = slugify_label(label)
    candidate = base
    suffix = 1
    while candidate in existing.values():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def extract_safe_label_from_stem(stem: str) -> str:
    """Extract the slug portion from a generated filename stem."""
    safe = stem.split("__", 1)[0]
    return safe.lower()


def is_windows_reserved(name: str) -> bool:
    stem = name
    if "." in stem:
        stem = stem.split(".", 1)[0]
    stem = stem.strip().upper()
    return stem in RESERVED_WINDOWS_NAMES


def load_class_label_map(dataset_dir: Path) -> Dict[str, str]:
    mapping_path = dataset_dir / "class_labels.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Expected class label map at {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping: Dict[str, str] = json.load(f)
    return {k.lower(): v for k, v in mapping.items()}


def save_class_label_map(mapping: Dict[str, str], path: Path) -> None:
    # sort by safe label for reproducibility
    sorted_mapping = {k: mapping[k] for k in sorted(mapping.keys())}
    path.write_text(json.dumps(sorted_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
"""Data preprocessing utilities for face detection, cropping, and alignment."""

from __future__ import annotations

from pathlib import Path


def preprocess_dataset(
    raw_dir: Path,
    processed_dir: Path,
    image_size: tuple[int, int] = (112, 112),
) -> Path:
    """Prepare raw face images into aligned crops in processed_dir.

    This scaffold ensures the output directory exists and documents where to
    plug in face detection, landmark alignment, and resizing.

    Args:
        raw_dir: Directory containing raw input images.
        processed_dir: Output directory for processed face crops.
        image_size: Target (width, height) for aligned face images.

    Returns:
        Path to the processed dataset directory.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[preprocess_dataset] raw_dir={raw_dir} "
        f"processed_dir={processed_dir} image_size={image_size}"
    )
    return processed_dir

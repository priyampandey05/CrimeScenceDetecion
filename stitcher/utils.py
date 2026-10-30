"""
Utility helpers for image loading, resizing, and validation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk as a BGR uint8 array.

    Returns None if the file cannot be read.
    """
    img = cv2.imread(str(path))
    return img


def resize_image(
    image: np.ndarray,
    max_dim: int = 2048,
) -> np.ndarray:
    """
    Resize *image* so that its longest dimension is at most *max_dim*.
    Preserves aspect ratio. Returns the original if already small enough.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def validate_images(images: List[np.ndarray]) -> Tuple[bool, str]:
    """
    Validate a list of images for stitching suitability.

    Returns:
        (is_valid, message)
    """
    if len(images) < 2:
        return False, "At least 2 images are required."

    for i, img in enumerate(images):
        if img is None or img.size == 0:
            return False, f"Image {i+1} is empty or could not be decoded."
        if len(img.shape) < 2:
            return False, f"Image {i+1} has an unexpected shape: {img.shape}."

    return True, "OK"


def images_to_thumbnails(
    images: List[np.ndarray],
    thumb_w: int = 200,
) -> List[np.ndarray]:
    """Return a list of thumbnail images (BGR) for quick preview."""
    thumbs = []
    for img in images:
        h, w = img.shape[:2]
        scale = thumb_w / w
        th = max(1, int(h * scale))
        thumbs.append(cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA))
    return thumbs


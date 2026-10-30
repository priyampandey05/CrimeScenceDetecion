"""
Cylindrical projection for panorama stitching.

For images captured by rotating a camera around its own vertical axis,
mapping to a cylinder converts rotation into pure horizontal translation.
This eliminates the wedge / trapezoidal distortion that sequential
homography stitching otherwise introduces.

Inverse mapping formula (cylinder coords → image coords):
    theta  = (x_cyl - cx) / f
    x_src  = f * tan(theta) + cx
    y_src  = (y_cyl - cy) / cos(theta) + cy
"""

import cv2
import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

def estimate_focal_length(image: np.ndarray, fov_deg: float = 60.0) -> float:
    """
    Estimate focal length in pixels for a given horizontal field-of-view.

    Args:
        image:   Input image (used for its width).
        fov_deg: Assumed horizontal FOV in degrees.  Typical values:
                    Wide-angle ≈ 75-90°  → f_factor ≈ 0.60–0.75
                    Normal lens ≈ 50-60° → f_factor ≈ 0.85–1.00
                    Telephoto  ≈ 30-40° → f_factor ≈ 1.50–2.00
    """
    w = image.shape[1]
    f = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return float(f)


def cylindrical_warp(img: np.ndarray, f: float) -> np.ndarray:
    """
    Project *img* onto a cylinder of focal length *f* (pixels).

    The output image has the same canvas size as the input; black pixels
    appear where the source image has no coverage (near the extreme left/
    right edges at high angles).

    Args:
        img: BGR uint8 source image.
        f:   Focal length in pixels.  Use :func:`estimate_focal_length`.

    Returns:
        Cylindrically-projected BGR image (same size as input).
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Vectorised inverse mapping: for each output pixel find its source
    xc = np.arange(w, dtype=np.float64) - cx   # shape (w,)
    yc = np.arange(h, dtype=np.float64) - cy   # shape (h,)
    X, Y = np.meshgrid(xc, yc)                 # both shape (h, w)

    theta     = X / f                           # horizontal angle
    cos_theta = np.cos(theta)

    map_x = (f * np.tan(theta) + cx).astype(np.float32)
    map_y = (Y / cos_theta + cy).astype(np.float32)

    warped = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped


def strip_black_cols(img: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    Remove fully-black columns from the left and right of *img*.
    Used after cylindrical warping to remove the curved edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_max = gray.max(axis=0)                  # max brightness per column
    valid = col_max > threshold
    if not valid.any():
        return img
    x0, x1 = np.where(valid)[0][[0, -1]]
    return img[:, x0 : x1 + 1]


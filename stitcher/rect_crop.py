"""
Largest-interior-rectangle crop utilities.

After stitching, the panorama canvas contains black (zero) regions wherever
no image contributed.  The find_largest_rectangle function uses an
O(W·H) histogram algorithm to find the biggest axis-aligned rectangle
composed entirely of valid (non-black) pixels.
"""

import cv2
import numpy as np
from typing import Tuple


def _largest_rect_in_histogram(heights: np.ndarray) -> Tuple[int, int, int]:
    """
    Return (x, width, height) of the largest rectangle that fits in
    a bar-chart represented by *heights* (1-D integer array).
    Stack-based O(n) algorithm.
    """
    n = len(heights)
    stack: list = []           # (start_x, bar_height)
    best = (0, 0, 0)           # (x, width, height)
    best_area = 0

    for i in range(n + 1):
        h = int(heights[i]) if i < n else 0
        start = i
        while stack and stack[-1][1] > h:
            sx, sh = stack.pop()
            w = i - sx
            area = w * sh
            if area > best_area:
                best_area = area
                best = (sx, w, sh)
            start = sx
        stack.append((start, h))

    return best


def find_largest_rectangle(
    mask: np.ndarray,
    erode_px: int = 2,
) -> Tuple[int, int, int, int]:
    """
    Find the largest axis-aligned rectangle whose pixels are all valid
    (non-zero) in *mask*.

    Args:
        mask:     Binary 2-D mask (uint8, 0 = invalid, >0 = valid).
        erode_px: Shrink the mask by this many pixels before searching
                  to avoid half-covered edge pixels.

    Returns:
        (x, y, w, h) of the best rectangle.
        Falls back to full image dimensions if nothing found.
    """
    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        mask = cv2.erode(mask, kernel)

    H, W = mask.shape
    bm = (mask > 0).astype(np.int32)

    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, W, 1)
    best_area = 0

    for row in range(H):
        # Accumulate column heights of consecutive valid rows
        heights = np.where(bm[row] > 0, heights + 1, 0)

        x, rw, rh = _largest_rect_in_histogram(heights)
        area = rw * rh
        if area > best_area:
            best_area = area
            best = (x, row - rh + 1, rw, rh)

    return best

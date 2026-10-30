"""
Image ordering utilities.
Builds a graph of image overlaps and sorts images into a stitching order.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2


class ImageOrderer:
    """
    Determines an optimal stitching order from pairwise match counts.
    Uses a greedy approach: start with the image pair with the most
    matches, then grow the sequence by appending the best-matching
    unvisited image at either end.
    """

    def __init__(self, n: int):
        """
        Args:
            n: Total number of images.
        """
        self.n = n
        # Adjacency matrix: match_count[i][j] = number of good matches
        self.match_count: Dict[Tuple[int, int], int] = {}

    def record_match(self, i: int, j: int, count: int) -> None:
        """Record pairwise match count (symmetric)."""
        key = (min(i, j), max(i, j))
        self.match_count[key] = count

    def get_match_count(self, i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        return self.match_count.get(key, 0)

    def compute_order(self) -> List[int]:
        """
        Return an ordered list of image indices for sequential stitching.
        Images with no valid matches are appended at the end.
        """
        if self.n == 1:
            return [0]

        # Find seed pair (most matches)
        best_pair = (0, 1)
        best_count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                c = self.get_match_count(i, j)
                if c > best_count:
                    best_count = c
                    best_pair = (i, j)

        order = list(best_pair)
        remaining = set(range(self.n)) - set(order)

        while remaining:
            # Try to extend at the right end
            best_right = None
            best_right_count = -1
            for idx in remaining:
                c = self.get_match_count(order[-1], idx)
                if c > best_right_count:
                    best_right_count = c
                    best_right = idx

            # Try to extend at the left end
            best_left = None
            best_left_count = -1
            for idx in remaining:
                c = self.get_match_count(order[0], idx)
                if c > best_left_count:
                    best_left_count = c
                    best_left = idx

            if best_right_count >= best_left_count and best_right is not None:
                order.append(best_right)
                remaining.remove(best_right)
            elif best_left is not None:
                order.insert(0, best_left)
                remaining.remove(best_left)
            else:
                # No matches — just append remaining in original order
                order.extend(sorted(remaining))
                break

        return order


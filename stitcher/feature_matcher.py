"""
Feature matching module.
Supports BFMatcher and FLANN-based matching with ratio-test filtering.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FeatureMatcher:
    """Matches keypoint descriptors between two images."""

    def __init__(
        self,
        method: str = "SIFT",
        ratio_thresh: float = 0.75,
        min_good_matches: int = 10,
    ):
        """
        Args:
            method: Feature method used ('SIFT' or 'ORB'). Controls
                    which underlying matcher is created.
            ratio_thresh: Lowe's ratio test threshold.
            min_good_matches: Minimum matches required to declare a
                              successful pairing.
        """
        self.ratio_thresh = ratio_thresh
        self.min_good_matches = min_good_matches
        self.method = method.upper()
        self._matcher = self._create_matcher()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _create_matcher(self):
        if self.method == "SIFT":
            # FLANN with KD-tree is faster for float descriptors (SIFT/SURF)
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BFMatcher with Hamming distance for binary descriptors (ORB)
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[cv2.DMatch]:
        """
        Match descriptors from two images using kNN + Lowe ratio test.

        Args:
            desc1: Descriptors from image 1.
            desc2: Descriptors from image 2.

        Returns:
            List of good DMatch objects after ratio filtering.
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        try:
            # ORB descriptors must be float32 for FLANN; keep uint8 for BF
            if self.method != "SIFT":
                d1 = desc1.astype(np.uint8)
                d2 = desc2.astype(np.uint8)
            else:
                d1 = desc1.astype(np.float32)
                d2 = desc2.astype(np.float32)

            raw = self._matcher.knnMatch(d1, d2, k=2)
        except cv2.error:
            return []

        good: List[cv2.DMatch] = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)

        return good

    def is_good_match(self, matches: List[cv2.DMatch]) -> bool:
        """Return True if the number of good matches meets the threshold."""
        return len(matches) >= self.min_good_matches

    def draw_matches(
        self,
        img1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_display: int = 50,
    ) -> np.ndarray:
        """Draw match lines between two images for visualisation."""
        display = matches[: min(max_display, len(matches))]
        vis = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            display,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return vis


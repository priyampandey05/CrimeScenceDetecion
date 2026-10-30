"""
Feature detection and description module.
Supports SIFT and ORB detectors with configurable parameters.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class FeatureDetector:
    """Handles feature detection and description for image stitching."""

    SUPPORTED_METHODS = ["SIFT", "ORB"]

    def __init__(self, method: str = "SIFT", max_features: int = 5000):
        """
        Initialize the feature detector.

        Args:
            method: Detection method - 'SIFT' or 'ORB'.
            max_features: Maximum number of features to detect.
        """
        self.method = method.upper()
        self.max_features = max_features

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Choose from {self.SUPPORTED_METHODS}."
            )

        self._detector = self._create_detector()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _create_detector(self):
        if self.method == "SIFT":
            return cv2.SIFT_create(nfeatures=self.max_features)
        else:  # ORB
            return cv2.ORB_create(nfeatures=self.max_features)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detect keypoints and compute descriptors for a single image.

        Args:
            image: BGR or grayscale image (uint8).

        Returns:
            Tuple of (keypoints, descriptors). Descriptors may be None if
            no keypoints were found.
        """
        gray = self._to_gray(image)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_method(self) -> str:
        return self.method


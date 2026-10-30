"""
Transformation estimation using RANSAC.
Supports Projective Homography (3x3) and Affine / Rigid (Euclidean) transformations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class HomographyEstimator:
    """Estimates geometric transformation between two sets of matched keypoints."""

    def __init__(
        self,
        ransac_reproj_thresh: float = 4.0,
        min_inliers: int = 8,
        transform_type: str = "affine",  # "affine", "homography"
    ):
        """
        Args:
            ransac_reproj_thresh: RANSAC reprojection-error threshold (pixels).
            min_inliers: Minimum inliers required for a valid transformation.
            transform_type: 'affine' (rigid similarity, preserves uniform height/width)
                            or 'homography' (full 8-DOF projective).
        """
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_inliers = min_inliers
        self.transform_type = transform_type.lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate transformation mapping image-2 coordinates into image-1 coordinates.

        Args:
            kp1: Keypoints in destination image (image 1).
            kp2: Keypoints in source image (image 2).
            matches: Good DMatch objects (queryIdx -> kp1, trainIdx -> kp2,
                     or queryIdx -> kp2, trainIdx -> kp1 depending on matcher).
                     Here we assume queryIdx -> kp1, trainIdx -> kp2.

        Returns:
            (H, mask) where H is 3x3 float64 matrix, or (None, None).
        """
        if len(matches) < 4:
            return None, None

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        if self.transform_type == "affine":
            # Estimate partial affine (translation + rotation + uniform scale, 4-DOF)
            # This strictly avoids the wedge/trapezoid distortion where width/height
            # expands toward the right.
            M, mask = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_thresh,
                maxIters=2000,
            )
            if M is None:
                return None, None

            inliers = int(mask.sum()) if mask is not None else 0
            if inliers < self.min_inliers:
                return None, None

            # Convert 2x3 affine matrix to 3x3 homogeneous matrix
            H = np.eye(3, dtype=np.float64)
            H[:2, :] = M
            return H, mask

        else:
            # Full 8-DOF Projective Homography
            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.ransac_reproj_thresh,
            )

            if H is None:
                return None, None

            inliers = int(mask.sum()) if mask is not None else 0
            if inliers < self.min_inliers:
                return None, None

            return H, mask

    def count_inliers(self, mask: Optional[np.ndarray]) -> int:
        """Return the number of inlier matches."""
        if mask is None:
            return 0
        return int(mask.sum())

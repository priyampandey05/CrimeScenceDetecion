"""
Image warping and multi-image blending utilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class ImageWarper:
    """Warps and blends images using transformation matrices."""

    def __init__(self, blend_mode: str = "multiband"):
        """
        Args:
            blend_mode: One of 'multiband', 'feather', 'simple'.
        """
        self.blend_mode = blend_mode.lower()

    # ------------------------------------------------------------------
    # Multi-Image Global Canvas Alignment & Blending
    # ------------------------------------------------------------------

    def blend_multiple(
        self,
        images: List[np.ndarray],
        transforms_to_ref: List[np.ndarray],
        crop_mode: str = "rect",
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Warp all images onto a single unified canvas using their transformations
        relative to the reference image, and blend them seamlessly.

        Args:
            images: List of BGR uint8 images.
            transforms_to_ref: List of 3x3 float64 matrices mapping each image
                               to the reference image coordinates.
            crop_mode: 'rect' (largest interior rectangle), 'tight', or 'none'.

        Returns:
            (blended_panorama, intermediate_accumulated_steps)
        """
        n = len(images)
        if n == 1:
            return images[0].copy(), [images[0].copy()]

        # 1. Compute bounding canvas dimensions across all warped images
        all_corners = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped_c = cv2.perspectiveTransform(corners, transforms_to_ref[i])
            all_corners.append(warped_c)

        all_corners = np.concatenate(all_corners, axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel()) - 2
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel()) + 2

        out_w = int(x_max - x_min)
        out_h = int(y_max - y_min)

        # Sanity clamping
        MAX_W, MAX_H = 16000, 8000
        out_w = max(50, min(out_w, MAX_W))
        out_h = max(50, min(out_h, MAX_H))

        # Translation matrix to shift the canvas origin into positive space (0, 0)
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

        # 2. Warp all images into the unified canvas
        warped_imgs: List[np.ndarray] = []
        masks: List[np.ndarray] = []

        for i, img in enumerate(images):
            H_canvas = T @ transforms_to_ref[i]
            warped = cv2.warpPerspective(img, H_canvas, (out_w, out_h))
            warped_imgs.append(warped)
            # Binary mask of valid pixels
            mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 1).astype(np.uint8)
            masks.append(mask)

        # 3. Create intermediate step views for visualization
        intermediates: List[np.ndarray] = []
        step_accum = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for i in range(n):
            valid = masks[i] > 0
            step_accum[valid] = warped_imgs[i][valid]
            # Crop to current content for clean viewing
            step_view = self.crop_to_valid(step_accum)
            intermediates.append(step_view)

        # 4. Blend images
        if self.blend_mode == "simple":
            blended = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            for i in range(n):
                valid = masks[i] > 0
                blended[valid] = warped_imgs[i][valid]
        elif self.blend_mode == "feather":
            blended = self._feather_blend_multiple(warped_imgs, masks, out_w, out_h)
        else:  # multiband
            blended = self._multiband_blend_multiple(warped_imgs, masks, out_w, out_h)

        # 5. Crop
        if crop_mode == "rect":
            blended = self.crop_largest_rectangle(blended)
        elif crop_mode == "tight":
            blended = self.crop_to_valid(blended)

        return blended, intermediates

    # ------------------------------------------------------------------
    # Pairwise Warping (Legacy / Incremental fallback)
    # ------------------------------------------------------------------

    def warp_and_blend(
        self,
        base: np.ndarray,
        src: np.ndarray,
        H: np.ndarray,
    ) -> np.ndarray:
        """Warp src into base's space and blend."""
        h1, w1 = base.shape[:2]
        h2, w2 = src.shape[:2]

        corners_src = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        corners_base = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners_src, H)
        all_corners = np.concatenate([corners_base, warped_corners], axis=0)

        x_min, y_min = np.int32(all_corners.min(axis=0).ravel()) - 1
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel()) + 1

        t = [-x_min, -y_min]
        H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float64)

        out_w = min(x_max - x_min, 12000)
        out_h = min(y_max - y_min, 6000)

        warped_src = cv2.warpPerspective(src, H_t @ H, (out_w, out_h))
        warped_base = cv2.warpPerspective(base, H_t, (out_w, out_h))

        return self._feather_blend_pair(warped_base, warped_src)

    # ------------------------------------------------------------------
    # Blending Implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _feather_blend_multiple(
        images: List[np.ndarray],
        masks: List[np.ndarray],
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """Distance-transform weighted multi-image blending."""
        accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_sum = np.zeros((out_h, out_w), dtype=np.float32)

        for img, mask in zip(images, masks):
            dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            accum += img.astype(np.float32) * dt[:, :, None]
            weight_sum += dt

        weight_sum[weight_sum == 0] = 1.0
        result = accum / weight_sum[:, :, None]
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _multiband_blend_multiple(
        images: List[np.ndarray],
        masks: List[np.ndarray],
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """
        Progressive multi-band Laplacian pyramid blending across all images.
        """
        try:
            # Pairwise pyramid blend incrementally across the images
            current = images[0].copy()
            current_mask = masks[0].copy()

            for i in range(1, len(images)):
                src = images[i]
                src_mask = masks[i]

                # Check overlap
                overlap = (current_mask > 0) & (src_mask > 0)
                if not overlap.any():
                    # No overlap: simple copy
                    only_src = src_mask > 0
                    current[only_src] = src[only_src]
                    current_mask = np.bitwise_or(current_mask, src_mask)
                    continue

                # Compute normalized distance-transform alpha weights
                dt_curr = cv2.distanceTransform(current_mask, cv2.DIST_L2, 5).astype(np.float32)
                dt_src = cv2.distanceTransform(src_mask, cv2.DIST_L2, 5).astype(np.float32)

                dt_total = dt_curr + dt_src
                dt_total[dt_total == 0] = 1.0
                alpha = dt_curr / dt_total
                alpha_3 = np.stack([alpha] * 3, axis=2)

                # Laplacian pyramid blend
                LEVELS = 4
                gp_curr = [current.astype(np.float32)]
                gp_src = [src.astype(np.float32)]
                gp_a = [alpha_3]

                for _ in range(LEVELS):
                    gp_curr.append(cv2.pyrDown(gp_curr[-1]))
                    gp_src.append(cv2.pyrDown(gp_src[-1]))
                    gp_a.append(cv2.pyrDown(gp_a[-1]))

                lp_curr = [gp_curr[LEVELS]]
                lp_src = [gp_src[LEVELS]]
                for lvl in range(LEVELS, 0, -1):
                    h, w = gp_curr[lvl - 1].shape[:2]
                    lp_curr.append(gp_curr[lvl - 1] - cv2.resize(cv2.pyrUp(gp_curr[lvl]), (w, h)))
                    lp_src.append(gp_src[lvl - 1] - cv2.resize(cv2.pyrUp(gp_src[lvl]), (w, h)))

                blended_pyramid = []
                for lc, ls, a in zip(lp_curr, lp_src, reversed(gp_a)):
                    ah, aw = a.shape[:2]
                    lh, lw = lc.shape[:2]
                    if ah != lh or aw != lw:
                        a = cv2.resize(a, (lw, lh))
                    blended_pyramid.append(lc * a + ls * (1.0 - a))

                res = blended_pyramid[0]
                for lvl in range(1, len(blended_pyramid)):
                    h, w = blended_pyramid[lvl].shape[:2]
                    res = blended_pyramid[lvl] + cv2.resize(cv2.pyrUp(res), (w, h))

                # Fill non-overlapping regions
                only_curr = (current_mask > 0) & (src_mask == 0)
                only_src = (src_mask > 0) & (current_mask == 0)
                for c in range(3):
                    res[:, :, c][only_curr] = current[:, :, c][only_curr]
                    res[:, :, c][only_src] = src[:, :, c][only_src]

                current = np.clip(res, 0, 255).astype(np.uint8)
                current_mask = np.bitwise_or(current_mask, src_mask)

            return current

        except Exception:
            return ImageWarper._feather_blend_multiple(images, masks, out_w, out_h)

    @staticmethod
    def _feather_blend_pair(base: np.ndarray, src: np.ndarray) -> np.ndarray:
        mask_base = (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) > 1).astype(np.uint8)
        mask_src = (cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) > 1).astype(np.uint8)

        dist_base = cv2.distanceTransform(mask_base, cv2.DIST_L2, 5)
        dist_src = cv2.distanceTransform(mask_src, cv2.DIST_L2, 5)

        dist_base_3 = np.stack([dist_base] * 3, axis=2)
        dist_src_3 = np.stack([dist_src] * 3, axis=2)
        total = dist_base_3 + dist_src_3
        total[total == 0] = 1

        w_base = dist_base_3 / total
        w_src = dist_src_3 / total

        result = base.astype(np.float32) * w_base + src.astype(np.float32) * w_src

        only_base = (mask_base > 0) & (mask_src == 0)
        only_src = (mask_src > 0) & (mask_base == 0)
        for c in range(3):
            result[:, :, c][only_base] = base.astype(np.float32)[:, :, c][only_base]
            result[:, :, c][only_src] = src.astype(np.float32)[:, :, c][only_src]

        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Post-processing Cropping
    # ------------------------------------------------------------------

    @staticmethod
    def crop_to_valid(image: np.ndarray) -> np.ndarray:
        """Crop to tight bounding box of non-zero pixels."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        return image[y : y + h, x : x + w]

    @staticmethod
    def crop_largest_rectangle(image: np.ndarray) -> np.ndarray:
        """
        Find and crop to the largest axis-aligned rectangle that lies entirely
        within valid pixels. Guarantees uniform width and height with no black edges.
        """
        from .rect_crop import find_largest_rectangle

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 1).astype(np.uint8) * 255

        x, y, w, h = find_largest_rectangle(mask, erode_px=2)

        if w < 50 or h < 50:
            return ImageWarper.crop_to_valid(image)

        return image[y : y + h, x : x + w]

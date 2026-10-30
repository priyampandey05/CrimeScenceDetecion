"""
Generate synthetic sample panorama images for testing.
Creates a wide scene and slices it into overlapping strips
with slight perspective perturbations so SIFT has real work to do.
"""

import cv2
import numpy as np
import os

# ── Reproducible randomness ───────────────────────────────────────────────────
rng = np.random.default_rng(42)

OUTPUT_DIR = "sample_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Build a rich synthetic scene ──────────────────────────────────────────────
def make_scene(w: int = 2400, h: int = 800) -> np.ndarray:
    scene = np.ones((h, w, 3), dtype=np.uint8) * 220  # light grey sky

    # Gradient sky
    for row in range(h // 2):
        t = row / (h // 2)
        scene[row, :] = [
            int(180 + 60 * (1 - t)),
            int(200 + 40 * (1 - t)),
            int(240),
        ]

    # Ground
    scene[h // 2 :, :] = [60, 110, 60]

    # ── Draw buildings ────────────────────────────────────────────────────────
    buildings = [
        (100, 250, 450, 50, (140, 140, 160)),
        (300, 180, 550, 50, (160, 150, 130)),
        (520, 300, 680, 50, (120, 130, 150)),
        (700, 200, 900, 50, (150, 140, 120)),
        (870, 280, 1050, 50, (130, 145, 155)),
        (1030, 160, 1250, 50, (145, 135, 125)),
        (1220, 260, 1420, 50, (135, 150, 140)),
        (1400, 220, 1620, 50, (125, 130, 145)),
        (1580, 300, 1760, 50, (150, 120, 130)),
        (1740, 180, 1960, 50, (140, 145, 150)),
        (1920, 240, 2100, 50, (130, 140, 135)),
        (2060, 190, 2280, 50, (145, 130, 150)),
    ]

    for x1, y1, x2, y2_base, color in buildings:
        # Building body
        cv2.rectangle(scene, (x1, y1), (x2, h // 2), color, -1)
        # Roof detail
        cv2.rectangle(scene, (x1, y1 - 15), (x2, y1), tuple(c + 20 for c in color), -1)
        # Windows
        win_w, win_h = 22, 18
        for wy in range(y1 + 20, h // 2 - 20, 38):
            for wx in range(x1 + 12, x2 - 12, 38):
                lit = rng.random() > 0.4
                wc = (255, 240, 180) if lit else (60, 60, 80)
                cv2.rectangle(scene, (wx, wy), (wx + win_w, wy + win_h), wc, -1)

    # ── Street & road markings ────────────────────────────────────────────────
    cv2.rectangle(scene, (0, h // 2 + 10), (w, h // 2 + 60), (70, 70, 70), -1)
    for x in range(50, w, 120):
        cv2.rectangle(scene, (x, h // 2 + 28), (x + 60, h // 2 + 42), (255, 255, 255), -1)

    # ── Trees ─────────────────────────────────────────────────────────────────
    tree_xs = list(range(80, w, 190))
    for tx in tree_xs:
        ty = h // 2 + 5
        trunk_h = rng.integers(40, 70)
        cv2.rectangle(scene, (tx - 5, ty - trunk_h), (tx + 5, ty), (80, 50, 30), -1)
        radius = rng.integers(28, 42)
        green = (
            int(rng.integers(30, 60)),
            int(rng.integers(100, 180)),
            int(rng.integers(20, 60)),
        )
        cv2.circle(scene, (tx, ty - int(trunk_h) - int(radius) + 10), int(radius), green, -1)
        cv2.circle(scene, (tx - 12, ty - int(trunk_h) - int(radius) + 20), max(1, int(radius) - 8), green, -1)
        cv2.circle(scene, (tx + 14, ty - int(trunk_h) - int(radius) + 18), max(1, int(radius) - 10), green, -1)

    # ── Clouds ────────────────────────────────────────────────────────────────
    cloud_specs = [
        (200, 100, 5), (600, 80, 4), (1000, 120, 5),
        (1400, 90, 4), (1800, 110, 5), (2200, 85, 4),
    ]
    for cx, cy, nb in cloud_specs:
        for _ in range(nb):
            ox = rng.integers(-40, 40)
            oy = rng.integers(-15, 15)
            r = rng.integers(30, 55)
            cv2.circle(scene, (cx + ox, cy + oy), r, (255, 255, 255), -1)

    # ── Signboards ────────────────────────────────────────────────────────────
    signs = [
        (400, 380, "CAFE", (0, 120, 220)),
        (900, 360, "BANK", (220, 100, 0)),
        (1350, 370, "SHOP", (50, 180, 50)),
        (1800, 355, "HOTEL", (180, 0, 120)),
    ]
    for sx, sy, text, col in signs:
        cv2.rectangle(scene, (sx - 5, sy - 5), (sx + 95, sy + 30), col, -1)
        cv2.putText(scene, text, (sx + 2, sy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Random feature dots (helps SIFT find corners) ─────────────────────────
    for _ in range(300):
        px = rng.integers(0, w)
        py = rng.integers(0, h)
        r = rng.integers(3, 8)
        col = tuple(int(c) for c in rng.integers(0, 200, 3).tolist())
        cv2.circle(scene, (px, py), r, col, -1)

    return scene


# ── Slice scene into overlapping strips with slight perspective warp ───────────
def slice_scene(scene: np.ndarray, n_strips: int = 4, overlap: float = 0.40):
    h, w = scene.shape[:2]
    strip_w = int(w / (n_strips - (n_strips - 1) * overlap))
    step = int(strip_w * (1 - overlap))

    strips = []
    for i in range(n_strips):
        x_start = i * step
        x_end = min(x_start + strip_w, w)
        strip = scene[:, x_start:x_end].copy()

        # Add slight perspective distortion to each strip
        sw = strip.shape[1]
        sh = strip.shape[0]
        jitter = 12
        src_pts = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
        dst_pts = src_pts + rng.uniform(-jitter, jitter, src_pts.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(strip, M, (sw, sh))
        strips.append(warped)

    return strips


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating synthetic scene …")
    scene = make_scene(2400, 800)

    # Save full scene for reference
    cv2.imwrite(os.path.join(OUTPUT_DIR, "full_scene_reference.jpg"), scene)
    print(f"  Full scene saved → {OUTPUT_DIR}/full_scene_reference.jpg")

    # Create 4 overlapping strips (shuffled to simulate random upload order)
    strips = slice_scene(scene, n_strips=4, overlap=0.40)
    order = [2, 0, 3, 1]   # shuffled order
    for out_idx, src_idx in enumerate(order):
        fname = os.path.join(OUTPUT_DIR, f"image_{out_idx + 1:02d}.jpg")
        cv2.imwrite(fname, strips[src_idx], [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Saved strip {src_idx} → {fname}")

    print(f"\nDone. Upload these 4 images to the Streamlit app:")
    for i in range(1, 5):
        print(f"  {OUTPUT_DIR}/image_{i:02d}.jpg")

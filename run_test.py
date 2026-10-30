"""
Headless test — runs the improved pipeline on provided images and reports results.
Usage:  python run_test.py [--dir sample_images]
"""

import cv2
import numpy as np
import os
import sys
import time
import glob
import math

sys.path.insert(0, os.path.dirname(__file__))
from stitcher.pipeline import StitchingPipeline

# ── Image source ──────────────────────────────────────────────────────────────
INPUT_DIR  = sys.argv[1] if len(sys.argv) > 1 else "sample_images"
OUTPUT_DIR = "sample_output_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
files = []
for e in exts:
    files += glob.glob(os.path.join(INPUT_DIR, e))
files = sorted(f for f in files if "reference" not in os.path.basename(f).lower())

print(f"Input dir : {INPUT_DIR}")
print(f"Found {len(files)} images:")
images = []
names  = []
for f in files:
    img = cv2.imread(f)
    if img is not None:
        images.append(img)
        names.append(os.path.basename(f))
        print(f"  {os.path.basename(f):35s}  {img.shape[1]}×{img.shape[0]}")

print()

# ── Run (cylindrical + affine uniform-width + rect-crop) ─────────────────────
fov_deg = 60
focal_px = images[0].shape[1] / (2 * math.tan(math.radians(fov_deg) / 2))

pipeline = StitchingPipeline(
    method="SIFT",
    max_features=5000,
    ratio_thresh=0.75,
    ransac_thresh=4.0,
    min_matches=10,
    min_inliers=8,
    transform_type="affine",   # Prevents wedge / trapezoid distortion!
    blend_mode="multiband",
    use_cylindrical=True,
    focal_length=focal_px,
    crop_mode="rect",
)

def cb(msg, frac):
    bar = "█" * int(frac * 32) + "░" * (32 - int(frac * 32))
    print(f"  [{bar}] {frac*100:5.1f}%  {msg}")

print("=" * 68)
print("  Stitching …")
print("=" * 68)
t0 = time.time()
result = pipeline.stitch(images, progress_cb=cb)
elapsed = time.time() - t0
print("=" * 68)
print(f"  Elapsed: {elapsed:.2f}s")
print("=" * 68)

# ── Save outputs ──────────────────────────────────────────────────────────────
if result.panorama is not None:
    out_path = os.path.join(OUTPUT_DIR, "panorama.jpg")
    cv2.imwrite(out_path, result.panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = result.panorama.shape[:2]
    print(f"\n✅ Panorama  →  {out_path}  ({w}×{h} px, ratio {w/h:.2f}:1)")
else:
    print("\n❌ Stitching failed.")

for i, inter in enumerate(result.intermediate_panoramas):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"step_{i:02d}.jpg"), inter,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

for pr in result.pair_results:
    if pr.match_vis is not None:
        a = names[pr.img_idx_a].replace(".jpg","")
        b = names[pr.img_idx_b].replace(".jpg","")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"match_{a}__{b}.jpg"), pr.match_vis)

print(f"\n📋 Pair statistics:")
print(f"  {'Pair':<50} {'Matches':>8} {'Inliers':>8} {'OK':>5}")
print("  " + "─" * 74)
for pr in result.pair_results:
    a = names[pr.img_idx_a]; b = names[pr.img_idx_b]
    ok = "✅" if pr.homography is not None else "❌"
    print(f"  {a} ↔ {b:<25} {pr.num_matches:>8} {pr.num_inliers:>8} {ok:>5}")

print(f"\n  Stitching order : {[names[i] for i in result.stitch_order]}")
print(f"  Failed images   : {[names[i] for i in result.failed_images] or 'None 🎉'}")

print(f"\n  Files in ./{OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
    print(f"    {f:<45} {kb:6} KB")

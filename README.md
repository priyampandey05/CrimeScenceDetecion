# 🔭 Panorama Stitching and Image Reconstruction System

An advanced Python and OpenCV-based image stitching system with an interactive Streamlit frontend. It automatically aligns and blends multiple overlapping photographs captured from different angles, rotations, and upload orders into a single, seamless, high-resolution panorama with **uniform height and width**.

---

## 🌟 Key Features & Improvements

### 1. Uniform Width & Height (Zero Wedge / Trumpet Distortion)
- **Problem**: Chaining standard 3×3 projective homographies causes projective division terms ($h_{20}, h_{21}$) to compound across multiple images, causing the right side of the canvas to flare outwards into a cone/wedge.
- **Solution**: 
  - **Cylindrical Projection**: Maps camera rotations into pure horizontal translations.
  - **Affine / Rigid Transformation Model (RANSAC)**: Enforces $h_{20}=0, h_{21}=0$, guaranteeing that horizontal lines stay horizontal and heights stay completely uniform across the entire panorama.
  - **Projective Homography Mode**: Still available via sidebar toggle for planar scenes.

### 2. Central Reference Frame Alignment
- Instead of compounding errors from left to right, the system designates the middle image of the sequence as the reference frame ($H_{ref} = I$).
- All images are warped directly to the global canvas in a single step, minimizing blur and symmetrically balancing alignment tolerances.

### 3. Smart Automatic Ordering
- Graph-based matching counts RANSAC inliers across all pairs and determines the true left-to-right sequence regardless of random upload order.

### 4. Seamless Multi-Band Blending
- **Multi-band Laplacian Pyramid**: Blends low-frequency illumination smoothly while retaining high-frequency edges and details without ghosting.
- **Distance-Transform Feathering**: Proportional alpha blending weighted by pixel distance to borders.
- **Simple Overlay**: Fast raw copy mode.

### 5. Automatic Clean Rectangular Cropping
- Implements an $O(W \cdot H)$ stack-based histogram algorithm to find the largest axis-aligned rectangle composed entirely of valid pixels, delivering crisp rectangular results without black borders.

### 6. Interactive Streamlit UI
- **Quick Sample Loaders**: 1-click loading of the 4 St. Petersburg river images or synthetic cityscape.
- **Visual Process Inspection**: Step-by-step intermediate panorama construction views.
- **Feature Correspondences**: Visual match lines and pairwise RANSAC inlier heatmap.
- **Export Options**: Download reconstructed panoramas in high-resolution PNG or JPEG.

---

## 📁 Project Structure

```
CrimeScenceDetecion/
├── app.py                    # Streamlit web application
├── run_test.py               # Headless CLI test runner
├── generate_samples.py       # Synthetic dataset generator
├── requirements.txt          # Python dependencies
├── run.sh                    # Startup script
├── real_images/              # 4 St. Petersburg scene images
├── sample_images/            # 4 Synthetic cityscape images
└── stitcher/
    ├── __init__.py
    ├── pipeline.py           # Core stitching orchestrator
    ├── cylindrical.py        # Cylindrical warping & focal length estimation
    ├── feature_detector.py   # SIFT & ORB feature detectors
    ├── feature_matcher.py    # FLANN / BFMatcher with Lowe's ratio test
    ├── homography.py         # RANSAC Affine & Homography estimators
    ├── ordering.py           # Graph-based image sequence optimizer
    ├── rect_crop.py          # Largest interior rectangle algorithm
    ├── warper.py             # Multi-image warping & Laplacian pyramid blending
    └── utils.py              # Image I/O & validation helpers
```

---

## 🚀 Quick Start

### 1. Activate the Virtual Environment

```bash
cd /Users/priyam/Desktop/CrimeScenceDetecion
source venv/bin/activate
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
# Or using the launcher script:
bash run.sh
```

Open your browser at **http://localhost:8501**.

### 3. Run Headless CLI Tests

```bash
# Test on real St. Petersburg images
venv/bin/python run_test.py real_images

# Test on synthetic cityscape
venv/bin/python run_test.py sample_images
```

---

## ⚙️ Settings Reference

| Parameter | Recommended Default | Description |
|-----------|--------------------|-------------|
| **Transformation Model** | `Affine` | `Affine` maintains uniform height/width. `Homography` enables full 8-DOF projective mapping. |
| **Cylindrical Projection** | `Enabled` | Projects 3D camera pan onto cylinder; converts rotation into translation. |
| **Field-of-View (FOV)** | `60°` | Estimated camera horizontal FOV (kit lens: 55–65°, wide-angle: 70–90°). |
| **Detector** | `SIFT` | `SIFT` (scale-invariant, accurate) or `ORB` (high-speed). |
| **Lowe's Ratio** | `0.75` | Distance ratio threshold for nearest-neighbor filtering. |
| **RANSAC Threshold** | `4.0 px` | Maximum reprojection error for inlier classification. |
| **Blending** | `Multiband` | Laplacian pyramid blending for seamless transitions. |
| **Cropping** | `Largest Rectangle` | Crops away black canvas corners for clean edges. |

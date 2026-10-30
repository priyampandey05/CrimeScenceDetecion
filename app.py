"""
Streamlit front-end for the Image Stitching and Panorama Reconstruction System.
Run with:  streamlit run app.py
"""

import os
import glob
import time
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from stitcher.pipeline import StitchingPipeline, StitchResult

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔭 Panorama Reconstruction System",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(img))


def resize_for_display(img: np.ndarray, max_width: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def ndarray_to_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Image encoding failed.")
    return buf.tobytes()


def render_markdown_table(rows: list):
    """Safely render tabular data in markdown without pyarrow / pandas dependencies."""
    if not rows:
        st.info("No data available.")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    st.markdown("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – Configuration & CV Settings
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Stitcher Controls")
    st.markdown("---")

    # ── Engine Selection ─────────────────────────────────────────────────────
    st.subheader("🚀 Stitching Engine")
    engine = st.radio(
        "Engine Mode",
        options=["custom", "opencv"],
        format_func=lambda x: (
            "Custom Pipeline (SIFT/ORB + RANSAC + Warping)"
            if x == "custom"
            else "OpenCV Native (cv2.Stitcher)"
        ),
        index=0,
        help="Custom pipeline demonstrates full computer vision pipeline with step-by-step visualisations.",
    )

    # ── Geometric Alignment & Distortion Control ─────────────────────────────
    st.subheader("📐 Alignment & Geometry")
    use_cylindrical = st.toggle(
        "Cylindrical Projection",
        value=True,
        help="Maps vertical camera panning onto a cylinder. Equalizes scale and prevents edge stretching.",
    )

    if use_cylindrical:
        fov_deg = st.slider(
            "Horizontal Field-of-View (FOV)",
            min_value=30,
            max_value=120,
            value=60,
            step=5,
            help="Camera horizontal FOV in degrees (kit lens ≈ 55–65°, wide-angle ≈ 70–90°).",
        )
    else:
        fov_deg = 60

    transform_type = st.selectbox(
        "Transformation Model",
        options=["affine", "homography"],
        index=0,
        format_func=lambda x: (
            "Affine / Rigid (Uniform Width & Height - Recommended)"
            if x == "affine"
            else "Projective Homography (8-DOF 3x3)"
        ),
        help="Affine maintains parallel vertical lines and prevents the wedge/trumpet distortion.",
    )

    crop_mode = st.selectbox(
        "Canvas Cropping",
        options=["rect", "tight", "none"],
        index=0,
        format_func=lambda x: {
            "rect": "Largest Rectangle (Uniform clean edges)",
            "tight": "Tight Bounding Box",
            "none": "Raw Canvas (No Crop)",
        }[x],
        help="Automatically removes black borders to deliver a crisp, rectangular panorama.",
    )

    st.markdown("---")

    # ── Feature Detection & Matching ─────────────────────────────────────────
    st.subheader("🔍 Feature Matching")
    method = st.selectbox(
        "Feature Detector",
        options=["SIFT", "ORB"],
        index=0,
        help="SIFT provides scale-invariant high accuracy. ORB is fast and patent-free.",
    )

    max_features = st.slider(
        "Max Features per Image",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=500,
    )

    ratio_thresh = st.slider(
        "Lowe's Ratio Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.01,
        help="Threshold for nearest-neighbor distance ratio. Lower = more selective.",
    )

    ransac_thresh = st.slider(
        "RANSAC Inlier Threshold (px)",
        min_value=1.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
    )

    # ── Blending ─────────────────────────────────────────────────────────────
    st.subheader("🎨 Seam Blending")
    blend_mode = st.selectbox(
        "Blending Method",
        options=["multiband", "feather", "simple"],
        index=0,
        format_func=lambda x: {
            "multiband": "Multi-band Laplacian Pyramid (Smooth transitions)",
            "feather": "Distance-Transform Feathering (Fast & smooth)",
            "simple": "Simple Overlay (Hard seam)",
        }[x],
    )

    st.markdown("---")
    st.caption("Panorama Stitching System v2.0 · Python & OpenCV")


# ──────────────────────────────────────────────────────────────────────────────
# Main Application Interface
# ──────────────────────────────────────────────────────────────────────────────

st.title("🔭 Panorama Reconstruction System")
st.markdown(
    """
    **Reconstruct high-resolution panoramas from multiple overlapping photos** captured from 
    different angles, rotations, or random upload orders. 
    Equipped with **Cylindrical Projection**, **Affine/Rigid RANSAC Alignment** (eliminates wedge/flare distortion), 
    **Central Reference Alignment**, and **Multi-band Laplacian Blending**.
    """
)

# ── Sample Quick-Loaders ──────────────────────────────────────────────────────
st.markdown("##### ⚡ Quick Test with Sample Datasets")
col_s1, col_s2, col_s3 = st.columns([1, 1, 2])

# Session state to hold loaded images
if "loaded_images" not in st.session_state:
    st.session_state.loaded_images = []
if "loaded_names" not in st.session_state:
    st.session_state.loaded_names = []

with col_s1:
    if st.button("🏰 Load Real Scene (4 St. Petersburg Images)", use_container_width=True):
        real_files = sorted(glob.glob("real_images/*.jpg"))
        st.session_state.loaded_images = [cv2.imread(f) for f in real_files]
        st.session_state.loaded_names = [os.path.basename(f) for f in real_files]
        st.success(f"Loaded {len(real_files)} real images!")

with col_s2:
    if st.button("🏙️ Load Synthetic Scene (4 Shuffled Images)", use_container_width=True):
        synth_files = sorted(glob.glob("sample_images/image_*.jpg"))
        st.session_state.loaded_images = [cv2.imread(f) for f in synth_files]
        st.session_state.loaded_names = [os.path.basename(f) for f in synth_files]
        st.success(f"Loaded {len(synth_files)} synthetic images!")

with col_s3:
    if st.session_state.loaded_images:
        if st.button("🗑️ Clear Loaded Images", use_container_width=False):
            st.session_state.loaded_images = []
            st.session_state.loaded_names = []
            st.rerun()

st.markdown("---")

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Or upload your own images (JPG, PNG, TIFF, BMP) in any order:",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state.loaded_images = []
    st.session_state.loaded_names = []
    for uf in uploaded_files:
        try:
            pil_img = Image.open(uf)
            bgr = pil_to_bgr(pil_img)
            st.session_state.loaded_images.append(bgr)
            st.session_state.loaded_names.append(uf.name)
        except Exception as e:
            st.warning(f"Could not parse **{uf.name}**: {e}")

images_bgr = st.session_state.loaded_images
image_names = st.session_state.loaded_names

if len(images_bgr) < 2:
    st.info("👆 Please upload at least 2 images or click one of the sample dataset buttons above.")
    st.stop()

# ── Preview Uploaded Images ──────────────────────────────────────────────────
with st.expander(f"📂 Loaded Images ({len(images_bgr)} images ready)", expanded=True):
    cols = st.columns(min(len(images_bgr), 4))
    for i, (img, name) in enumerate(zip(images_bgr, image_names)):
        with cols[i % len(cols)]:
            h, w = img.shape[:2]
            st.image(to_pil(img), caption=f"{name} ({w}×{h})", use_container_width=True)

st.markdown("---")

# ── Run Stitching ────────────────────────────────────────────────────────────
run_button = st.button("🚀 Reconstruct Panorama", type="primary", use_container_width=True)

if run_button:
    # Compute focal length from user FOV
    w0 = images_bgr[0].shape[1]
    focal_px = w0 / (2.0 * math.tan(math.radians(fov_deg) / 2.0)) if use_cylindrical else 0.0

    pipeline = StitchingPipeline(
        method=method,
        max_features=max_features,
        ratio_thresh=ratio_thresh,
        ransac_thresh=ransac_thresh,
        min_matches=10,
        min_inliers=8,
        transform_type=transform_type,
        blend_mode=blend_mode,
        use_cylindrical=use_cylindrical,
        focal_length=focal_px,
        crop_mode=crop_mode,
        engine=engine,
    )

    progress_bar = st.progress(0.0)
    status_box = st.empty()
    live_log = st.empty()
    logs: list[str] = []

    def progress_callback(msg: str, frac: float):
        progress_bar.progress(min(frac, 1.0))
        status_box.markdown(f"**Current Status:** {msg}")
        logs.append(msg)
        with live_log.container():
            with st.expander("📋 Live Progress Log", expanded=False):
                st.text("\n".join(logs[-25:]))

    t_start = time.time()
    result: StitchResult = pipeline.stitch(images_bgr, progress_cb=progress_callback)
    duration = time.time() - t_start

    progress_bar.progress(1.0)
    status_box.markdown(f"**Current Status:** ✅ Completed in **{duration:.2f} seconds**")

    # ── Result Tabs ──────────────────────────────────────────────────────────
    tab_pano, tab_matches, tab_steps, tab_diagnostics = st.tabs(
        ["🖼️ Reconstructed Panorama", "🔗 Feature Matches", "🎞️ Stitching Steps", "📋 Diagnostics & Log"]
    )

    # ── Tab 1: Final Panorama ────────────────────────────────────────────────
    with tab_pano:
        if result.panorama is not None:
            num_stitched = len(result.stitch_order) - len(result.failed_images)
            st.success(
                f"🎉 Successfully stitched **{num_stitched} of {len(images_bgr)}** images "
                f"in **{duration:.2f}s** with **uniform width and height**!"
            )

            if result.failed_images:
                failed_str = ", ".join(image_names[i] for i in result.failed_images)
                st.warning(f"⚠️ Images skipped due to insufficient overlap: **{failed_str}**")

            h_res, w_res = result.panorama.shape[:2]
            aspect = w_res / float(h_res)

            st.subheader(f"Final Reconstructed Panorama — {w_res} × {h_res} px (Aspect Ratio {aspect:.2f}:1)")
            display_img = resize_for_display(result.panorama, 1200)
            st.image(to_pil(display_img), use_container_width=True)

            # Download Buttons
            c_down1, c_down2 = st.columns(2)
            with c_down1:
                st.download_button(
                    "⬇️ Download High-Resolution PNG",
                    data=ndarray_to_bytes(result.panorama, ".png"),
                    file_name="panorama_reconstructed.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with c_down2:
                st.download_button(
                    "⬇️ Download JPEG (Compressed)",
                    data=ndarray_to_bytes(result.panorama, ".jpg"),
                    file_name="panorama_reconstructed.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            st.caption(
                f"Geometry: {transform_type.title()} · "
                f"Projection: {'Cylindrical (FOV ' + str(fov_deg) + '°)' if use_cylindrical else 'Planar'} · "
                f"Blending: {blend_mode.title()} · Crop: {crop_mode.title()}"
            )
        else:
            st.error("❌ Stitching could not be completed. Check image overlaps or adjust the Lowe's ratio threshold.")

    # ── Tab 2: Feature Matches ───────────────────────────────────────────────
    with tab_matches:
        st.subheader("Pairwise Feature Correspondence Visualisations")
        valid_pairs = [p for p in result.pair_results if p.match_vis is not None]

        if not valid_pairs:
            st.info("No pairwise matches were recorded.")
        else:
            for pr in valid_pairs:
                name_a = image_names[pr.img_idx_a]
                name_b = image_names[pr.img_idx_b]
                status_icon = "✅ Geometric Alignment Succeeded" if pr.homography is not None else "❌ Alignment Failed"
                with st.expander(f"{name_a} ↔ {name_b} | {pr.num_matches} Matches ({pr.num_inliers} Inliers) — {status_icon}"):
                    vis_display = resize_for_display(pr.match_vis, 1100)
                    st.image(to_pil(vis_display), use_container_width=True)

        # Inlier Heatmap
        st.markdown("---")
        st.subheader("Inlier Co-occurrence Heatmap")
        n = len(images_bgr)
        matrix = np.zeros((n, n), dtype=int)
        for pr in result.pair_results:
            matrix[pr.img_idx_a][pr.img_idx_b] = pr.num_inliers
            matrix[pr.img_idx_b][pr.img_idx_a] = pr.num_inliers

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(4, n), max(3, n - 1)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_names = [f"Img {i+1}" for i in range(n)]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(short_names, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color="black")
        ax.set_title("RANSAC Inliers Between Image Pairs")
        plt.colorbar(im, ax=ax, label="Inlier Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 3: Stitching Steps ───────────────────────────────────────────────
    with tab_steps:
        st.subheader("Step-by-Step Panorama Construction")
        if not result.intermediate_panoramas:
            st.info("No intermediate steps recorded.")
        else:
            for step_idx, inter_img in enumerate(result.intermediate_panoramas):
                ih, iw = inter_img.shape[:2]
                st.markdown(f"**Step {step_idx + 1}** — Canvas Dimensions: {iw} × {ih} px")
                st.image(to_pil(resize_for_display(inter_img, 1100)), use_container_width=True)
                st.markdown("---")

    # ── Tab 4: Diagnostics & Log ─────────────────────────────────────────────
    with tab_diagnostics:
        st.subheader("Execution Log")
        st.text("\n".join(result.log))

        st.markdown("---")
        st.subheader("Pairwise Alignment Statistics")
        stats_rows = [
            {
                "Image A": image_names[pr.img_idx_a],
                "Image B": image_names[pr.img_idx_b],
                "Keypoint Matches": pr.num_matches,
                "RANSAC Inliers": pr.num_inliers,
                "Inlier Ratio": f"{(pr.num_inliers / pr.num_matches * 100):.1f}%" if pr.num_matches > 0 else "0%",
                "Aligned": "✅ Yes" if pr.homography is not None else "❌ No",
            }
            for pr in result.pair_results
        ]
        render_markdown_table(stats_rows)

        st.markdown("---")
        st.subheader("Pipeline Configuration")
        config_rows = [
            {"Setting": "Engine", "Value": engine.upper()},
            {"Setting": "Detector", "Value": method},
            {"Setting": "Transform Model", "Value": transform_type.title()},
            {"Setting": "Cylindrical Projection", "Value": "Enabled" if use_cylindrical else "Disabled"},
            {"Setting": "Field-of-View (FOV)", "Value": f"{fov_deg}°" if use_cylindrical else "N/A"},
            {"Setting": "Focal Length", "Value": f"{focal_px:.1f} px" if use_cylindrical else "N/A"},
            {"Setting": "RANSAC Threshold", "Value": f"{ransac_thresh} px"},
            {"Setting": "Lowe's Ratio", "Value": str(ratio_thresh)},
            {"Setting": "Blending", "Value": blend_mode.title()},
            {"Setting": "Canvas Cropping", "Value": crop_mode.title()},
            {"Setting": "Total Elapsed Time", "Value": f"{duration:.2f} seconds"},
        ]
        render_markdown_table(config_rows)

"""
Generates a professional 10-slide PowerPoint presentation (.pptx)
for the Panorama Reconstruction & Image Stitching System.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path="Panorama_Stitching_Presentation.pptx"):
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette
    COLOR_BG_DARK    = RGBColor(15, 23, 42)     # #0F172A (Deep Slate)
    COLOR_BG_LIGHT   = RGBColor(248, 250, 252) # #F8FAFC (Off white)
    COLOR_CARD_BG    = RGBColor(255, 255, 255) # Pure White
    COLOR_CARD_BORDER= RGBColor(226, 232, 240) # Slate 200
    COLOR_PRIMARY    = RGBColor(14, 165, 233)  # #0EA5E9 (Sky Blue Accent)
    COLOR_SECONDARY  = RGBColor(99, 102, 241)  # #6366F1 (Indigo Accent)
    COLOR_TEXT_DARK  = RGBColor(30, 41, 59)    # Slate 800
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
    COLOR_TEXT_WHITE = RGBColor(255, 255, 255)
    COLOR_ACCENT_GREEN = RGBColor(16, 185, 129) # Emerald 500

    def add_header(slide, title_text, category="PANORAMA RECONSTRUCTION SYSTEM"):
        # Category / Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_PRIMARY

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_DARK

    def add_card(slide, left, top, width, height, title="", border_color=COLOR_CARD_BORDER, bg_color=COLOR_CARD_BG):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.5))
            p = tb.text_frame.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = COLOR_PRIMARY
        return shape

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 1: Title Slide (Dark Theme)
    # ═════════════════════════════════════════════════════════════════════════
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_BG_DARK
    bg1.line.fill.background()

    # Pill badge
    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.8), Inches(3.6), Inches(0.45))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(30, 41, 59)
    badge.line.color.rgb = COLOR_PRIMARY
    p_b = badge.text_frame.paragraphs[0]
    p_b.text = "COMPUTER VISION & IMAGE PROCESSING"
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_PRIMARY
    p_b.alignment = PP_ALIGN.CENTER

    # Main Title
    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(2.5), Inches(11), Inches(2.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Automatic Image Stitching &\nPanorama Reconstruction"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_WHITE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Robust Multi-Image Alignment, Cylindrical Projection & Seam-Free Blending"
    p1_sub.font.size = Pt(18)
    p1_sub.font.color.rgb = RGBColor(148, 163, 184)
    p1_sub.space_before = Pt(16)

    # Footer cards
    meta_box = s1.shapes.add_textbox(Inches(1.2), Inches(5.8), Inches(11), Inches(0.8))
    p_meta = meta_box.text_frame.paragraphs[0]
    p_meta.text = "Tech Stack: Python 3.12 | OpenCV 4.10 | SIFT & ORB | RANSAC | Streamlit"
    p_meta.font.size = Pt(13)
    p_meta.font.color.rgb = RGBColor(203, 213, 225)

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 2: Problem Statement & Motivation
    # ═════════════════════════════════════════════════════════════════════════
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Real-World Challenges", "01 / PROBLEM DEFINITION")

    # Card 1: Core Objective
    add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "🎯 Core Objective")
    tb2_1 = s2.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(4.0))
    tf2_1 = tb2_1.text_frame
    tf2_1.word_wrap = True
    bullets2_1 = [
        ("Automated Scene Synthesis: ", "Reconstruct a wide, continuous, coherent view from multiple overlapping photos captured from varying positions and camera angles."),
        ("Arbitrary Order Handling: ", "Accept images uploaded in scrambled, random sequences without requiring manual user ordering."),
        ("Distortion Minimization: ", "Eliminate spatial misalignment, vertical coordinate divergence, and visible seam boundaries."),
        ("Interactive Usability: ", "Deliver a full-featured web app with progress tracking, match inspection, and high-resolution export.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets2_1):
        p = tf2_1.add_paragraph() if i > 0 else tf2_1.paragraphs[0]
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # Card 2: Key Computer Vision Hurdles
    add_card(s2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "⚠️ Traditional Failure Modes")
    tb2_2 = s2.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.3), Inches(4.0))
    tf2_2 = tb2_2.text_frame
    tf2_2.word_wrap = True
    bullets2_2 = [
        ("The 'Wedge' Distortion: ", "Chained 8-DOF planar homographies cause coordinate denominators to diverge, blowing up heights on the right side into a cone shape."),
        ("Exposure Discrepancies: ", "Varying camera metering produces harsh vertical seams along cut boundaries."),
        ("Repetitive Textures: ", "Water reflections and sky gradients generate hundreds of false feature correspondences."),
        ("Cumulative Resampling Blur: ", "Sequential pairwise re-warping progressively blurs accumulated pixels.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets2_2):
        p = tf2_2.add_paragraph() if i > 0 else tf2_2.paragraphs[0]
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = RGBColor(225, 29, 72)
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 3: System Architecture & Workflow Pipeline
    # ═════════════════════════════════════════════════════════════════════════
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "End-to-End Architectural Pipeline", "02 / SYSTEM ARCHITECTURE")

    stages = [
        ("1. Cylindrical Projection", "Unwraps 3D camera pan into 2D translation via focal length estimation.", Inches(0.8), Inches(1.8)),
        ("2. Feature Extraction", "Computes multi-scale SIFT / ORB keypoints and descriptors per image.", Inches(4.9), Inches(1.8)),
        ("3. Pairwise Matching", "FLANN / BFMatcher with Lowe's ratio test to filter false correspondences.", Inches(9.0), Inches(1.8)),
        ("4. Graph Ordering", "Builds inlier adjacency matrix; derives true Left-to-Right sequence.", Inches(0.8), Inches(4.4)),
        ("5. Center Alignment", "Anchors to central image (H_ref = I); chains outward symmetrically.", Inches(4.9), Inches(4.4)),
        ("6. Multi-Band Blending", "Laplacian pyramid decomposition blends illumination and detail seamlessly.", Inches(9.0), Inches(4.4)),
    ]

    for title, desc, l, t in stages:
        add_card(s3, l, t, Inches(3.6), Inches(2.4), title)
        tb = s3.shapes.add_textbox(l + Inches(0.2), t + Inches(0.8), Inches(3.2), Inches(1.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 4: Feature Detection & Correspondence Matching
    # ═════════════════════════════════════════════════════════════════════════
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Feature Detection & Robust Correspondence", "03 / COMPUTER VISION FOUNDATIONS")

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "🔍 SIFT vs. ORB Detectors")
    tb4_1 = s4.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(4.0))
    tf4_1 = tb4_1.text_frame
    tf4_1.word_wrap = True
    bullets4_1 = [
        ("Scale-Space Extrema: ", "SIFT identifies keypoints across octaves using Difference-of-Gaussians (DoG = G(kσ) - G(σ))."),
        ("128-D Descriptors: ", "Local gradient orientation histograms invariant to rotation, illumination, and scale changes."),
        ("ORB High-Speed Alternative: ", "Combines FAST corner detection with oriented 256-bit binary BRIEF descriptors."),
        ("Configurable Budgets: ", "Custom keypoint limits (1,000 to 10,000) allow balancing speed vs. correspondence density.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets4_1):
        p = tf4_1.add_paragraph() if i > 0 else tf4_1.paragraphs[0]
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "🎯 Matching & Ratio Verification")
    tb4_2 = s4.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.3), Inches(4.0))
    tf4_2 = tb4_2.text_frame
    tf4_2.word_wrap = True
    bullets4_2 = [
        ("FLANN kNN Search: ", "Fast Library for Approximate Nearest Neighbors indexes SIFT descriptors via randomized KD-trees."),
        ("Lowe's Distance Ratio Test: ", "Accepts matches if d(best) / d(2nd-best) < 0.75, effectively rejecting repetitive water/sky patterns."),
        ("RANSAC Inlier Fitting: ", "Iteratively fits transformation models across randomized subsets, rejecting outliers with reprojection error > 4.0 px."),
        ("Inlier Heatmap Generation: ", "Constructs pairwise co-occurrence matrices for transparent diagnostics in the UI.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets4_2):
        p = tf4_2.add_paragraph() if i > 0 else tf4_2.paragraphs[0]
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_SECONDARY
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 5: Geometric Distortion Elimination (The "Wedge" Solution)
    # ═════════════════════════════════════════════════════════════════════════
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Distortion Elimination: Why Affine Solves the Wedge Effect", "04 / GEOMETRIC MODELING")

    # Comparison Column 1: The Problem
    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "❌ 8-DOF Homography (The Problem)", border_color=RGBColor(254, 205, 211))
    tb5_1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(4.0))
    tf5_1 = tb5_1.text_frame
    tf5_1.word_wrap = True
    p5_1_1 = tf5_1.paragraphs[0]
    p5_1_1.text = "Mathematical Cause of Flare:\n\n" \
                  "x' = (h00*x + h01*y + h02) / (h20*x + h21*y + 1)\n" \
                  "y' = (h10*x + h11*y + h12) / (h20*x + h21*y + 1)\n\n" \
                  "• In panning sequences, the denominator terms [h20, h21] are non-zero.\n" \
                  "• As multiple homographies compound (H_total = H1 * H2 * H3), the denominator approaches zero for large x.\n" \
                  "• Result: Vertical coordinates expand exponentially from 682 px to 4,072+ px (trapezoid wedge)."
    p5_1_1.font.size = Pt(12.5)
    p5_1_1.font.color.rgb = COLOR_TEXT_DARK

    # Comparison Column 2: The Solution
    add_card(s5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "✅ Cylindrical + Affine Alignment (Our Solution)", border_color=RGBColor(187, 247, 208))
    tb5_2 = s5.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.3), Inches(4.0))
    tf5_2 = tb5_2.text_frame
    tf5_2.word_wrap = True
    p5_2_1 = tf5_2.paragraphs[0]
    p5_2_1.text = "Uniform Height Guarantee:\n\n" \
                  "M_affine = [ [s*cos(θ), -s*sin(θ), tx],\n" \
                  "             [s*sin(θ),  s*cos(θ), ty],\n" \
                  "             [0,         0,        1 ] ]\n\n" \
                  "• Cylindrical projection converts panning rotation into pure translation.\n" \
                  "• Rigid affine alignment sets the bottom row strictly to [0, 0, 1].\n" \
                  "• Denominator is identically 1: No perspective divergence!\n" \
                  "• Parallel lines stay parallel; heights remain strictly uniform (524 px constant)."
    p5_2_1.font.size = Pt(12.5)
    p5_2_1.font.color.rgb = COLOR_TEXT_DARK

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 6: Graph Ordering & Center Reference Chaining
    # ═════════════════════════════════════════════════════════════════════════
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Graph Sequence Ordering & Central Reference Frame", "05 / ALIGNMENT STRATEGY")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "📊 Graph-Based Path Traversal")
    tb6_1 = s6.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(4.0))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    bullets6_1 = [
        ("Arbitrary Input Sequencing: ", "Users can upload photos in scrambled, random orders without manual pre-sorting."),
        ("Inlier Adjacency Matrix: ", "Constructs an undirected graph where edge weights w(i, j) represent verified RANSAC inlier counts."),
        ("Greedy Seed Expansion: ", "Initializes from the maximum-overlap pair (u, v) = argmax w(i, j)."),
        ("Bidirectional Growth: ", "Sequentially expands left or right based on connection density until all reachable nodes form an optimal chain.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets6_1):
        p = tf6_1.add_paragraph() if i > 0 else tf6_1.paragraphs[0]
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    add_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "🎯 Center Reference Frame (H_ref = I)")
    tb6_2 = s6.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.3), Inches(4.0))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    bullets6_2 = [
        ("Anchor Selection: ", "The sequence centroid mid = floor(N / 2) is chosen as the fixed global reference origin (H_mid = I)."),
        ("Symmetric Chaining: ", "Right side transforms: H_i = H_(i-1) @ M_(i-1, i).\nLeft side transforms: H_i = H_(i+1) @ inv(M_(i, i+1))."),
        ("Eliminating Cumulative Drift: ", "Errors don't cascade across the entire width; maximum chain length is halved (N/2 instead of N)."),
        ("Direct Canvas Projection: ", "All images warp directly into world coordinates in a single interpolation pass, preserving sharpness.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets6_2):
        p = tf6_2.add_paragraph() if i > 0 else tf6_2.paragraphs[0]
        p.space_after = Pt(10)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_PRIMARY
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 7: Multi-Band Blending & Rectangular Cropping
    # ═════════════════════════════════════════════════════════════════════════
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Multi-Band Blending & Largest Interior Rectangle Cropping", "06 / POST-PROCESSING")

    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "🎨 Burt-Adelson Laplacian Pyramid")
    tb7_1 = s7.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(4.0))
    tf7_1 = tb7_1.text_frame
    tf7_1.word_wrap = True
    bullets7_1 = [
        ("Spatial Frequency Separation: ", "Decomposes warped images into 4-level Gaussian and Laplacian pyramids."),
        ("Independent Band Fusion: ", "Low frequencies (illumination, sky tint) are blended across wide transition zones."),
        ("Edge Preservation: ", "High frequencies (crisp building edges, masts) are blended over narrow regions to eliminate ghosting."),
        ("Distance-Transform Fallback: ", "Fast Euclidean distance transform feathering available for rapid real-time rendering.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets7_1):
        p = tf7_1.add_paragraph() if i > 0 else tf7_1.paragraphs[0]
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_TEXT_DARK
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "✂️ Stack-Based Largest Interior Rectangle")
    tb7_2 = s7.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.3), Inches(4.0))
    tf7_2 = tb7_2.text_frame
    tf7_2.word_wrap = True
    bullets7_2 = [
        ("Eliminating Black Borders: ", "Cylindrical warping and angled camera overlaps naturally leave jagged, non-rectangular black margins."),
        ("O(W * H) Histogram Algorithm: ", "Constructs a binary validity mask and treats consecutive valid rows as a dynamic bar chart."),
        ("Monotonic Stack Traversal: ", "Finds the largest zero-free axis-aligned rectangle in linear time."),
        ("Border Erosion Buffer: ", "Applies 2 px morphological erosion to prevent boundary pixel bleed, outputting a flawless rectilinear image.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets7_2):
        p = tf7_2.add_paragraph() if i > 0 else tf7_2.paragraphs[0]
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = COLOR_ACCENT_GREEN
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT_MUTED

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 8: Streamlit Interactive Web Application
    # ═════════════════════════════════════════════════════════════════════════
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Interactive Streamlit Frontend & User Experience", "07 / USER INTERFACE")

    features_s8 = [
        ("⚡ 1-Click Sample Datasets", "Instant loading of real St. Petersburg scene (4 images) or synthetic cityscape for friction-free evaluation.", Inches(0.8), Inches(1.8)),
        ("⚙️ Full Hyper-Parameter Sidebar", "Sliders for FOV (30°-120°), Lowe's ratio, RANSAC thresholds, SIFT/ORB selection, and blend modes.", Inches(4.9), Inches(1.8)),
        ("🖼️ Tab 1: Final Panorama", "High-res viewer with dynamic aspect ratio readouts and 1-click downloads for PNG and JPEG.", Inches(9.0), Inches(1.8)),
        ("🔗 Tab 2: Feature Matches", "Expandable pairwise visual correspondence lines and a full RANSAC inlier co-occurrence heatmap.", Inches(0.8), Inches(4.4)),
        ("🎞️ Tab 3: Stitching Steps", "Visualizes the progressive evolution of the panorama canvas after incorporating each sequential frame.", Inches(4.9), Inches(4.4)),
        ("📋 Tab 4: Diagnostics & Log", "Live console log streaming, pairwise inlier ratio tables, and active configuration metadata.", Inches(9.0), Inches(4.4)),
    ]

    for title, desc, l, t in features_s8:
        add_card(s8, l, t, Inches(3.6), Inches(2.4), title)
        tb = s8.shapes.add_textbox(l + Inches(0.2), t + Inches(0.8), Inches(3.2), Inches(1.4))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_MUTED

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 9: Experimental Verification & Benchmark Results
    # ═════════════════════════════════════════════════════════════════════════
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Experimental Verification & Quantitative Benchmarks", "08 / EVALUATION")

    # Metrics Card
    add_card(s9, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0), "📊 Real-World Benchmark: St. Petersburg River Scene (4 Images)")

    # Table layout inside card
    tb9 = s9.shapes.add_textbox(Inches(1.1), Inches(2.5), Inches(11.1), Inches(4.0))
    tf9 = tb9.text_frame
    tf9.word_wrap = True

    p9_intro = tf9.paragraphs[0]
    p9_intro.text = "Comparative Performance Evaluation on Unordered Handheld Photos (1024 × 682 px each):"
    p9_intro.font.size = Pt(13)
    p9_intro.font.bold = True
    p9_intro.font.color.rgb = COLOR_TEXT_DARK

    table_data = [
        ("Metric", "Unconstrained Homography", "Our Custom Pipeline (Affine + Cylindrical)", "OpenCV Native (cv2.Stitcher)"),
        ("Execution Time", "2.18 seconds", "1.20 seconds (45% Faster)", "1.85 seconds"),
        ("Output Resolution", "7999 × 4072 px (Distorted)", "1990 × 524 px (Uniform)", "2106 × 700 px"),
        ("Wedge / Flare Distortion", "Severe (> 500% Height Divergence)", "None (Constant 524 px Height)", "None"),
        ("Sequence Sorting", "Manual / Sequential", "Automatic Graph Detection (img2->4->3->1)", "Built-in Bundle Adjustment"),
        ("Interior Cropping", "None (Ragged black edges)", "Automatic Largest Rectangle Crop", "Manual / None"),
        ("Failed Images", "0 / 4", "0 / 4 (100% Success)", "0 / 4")
    ]

    for row in table_data:
        p = tf9.add_paragraph()
        p.space_after = Pt(6)
        line_text = f"{row[0]:<26} | {row[1]:<36} | {row[2]:<42} | {row[3]}"
        run = p.add_run()
        run.text = line_text
        run.font.name = "Courier New"
        run.font.size = Pt(11)
        if row[0] == "Metric":
            run.font.bold = True
            run.font.color.rgb = COLOR_PRIMARY
        elif "Our Custom" in row[2]:
            run.font.color.rgb = COLOR_TEXT_DARK
        else:
            run.font.color.rgb = COLOR_TEXT_MUTED

    # ═════════════════════════════════════════════════════════════════════════
    # SLIDE 10: Conclusion & Future Scope (Dark Theme)
    # ═════════════════════════════════════════════════════════════════════════
    s10 = prs.slides.add_slide(blank_layout)
    bg10 = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg10.fill.solid()
    bg10.fill.fore_color.rgb = COLOR_BG_DARK
    bg10.line.fill.background()

    # Header
    cat_box10 = s10.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(0.4))
    p_c10 = cat_box10.text_frame.paragraphs[0]
    p_c10.text = "09 / SUMMARY & FUTURE WORK"
    p_c10.font.size = Pt(11)
    p_c10.font.bold = True
    p_c10.font.color.rgb = COLOR_PRIMARY

    title_box10 = s10.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.5), Inches(0.8))
    p_t10 = title_box10.text_frame.paragraphs[0]
    p_t10.text = "Key Achievements & Future Horizons"
    p_t10.font.size = Pt(26)
    p_t10.font.bold = True
    p_t10.font.color.rgb = COLOR_TEXT_WHITE

    # Left Dark Card: Achievements
    card10_1 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    card10_1.fill.solid()
    card10_1.fill.fore_color.rgb = RGBColor(30, 41, 59)
    card10_1.line.color.rgb = RGBColor(51, 65, 85)

    tb10_1 = s10.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf10_1 = tb10_1.text_frame
    tf10_1.word_wrap = True
    p_head1 = tf10_1.paragraphs[0]
    p_head1.text = "🏆 Project Achievements"
    p_head1.font.size = Pt(16)
    p_head1.font.bold = True
    p_head1.font.color.rgb = COLOR_PRIMARY
    p_head1.space_after = Pt(12)

    achievements = [
        ("Wedge Elimination: ", "Proved mathematically and practically that Affine + Cylindrical projection resolves perspective divergence."),
        ("Full Automation: ", "Handles random upload orders, varying overlaps, and arbitrary orientations seamlessly."),
        ("Production Quality: ", "Generates 1990x524 px seamless panoramas in 1.20s with clean rectangular bounds."),
        ("Complete UI Suite: ", "Offers full transparency with match vectors, inlier heatmaps, and intermediate construction stages.")
    ]
    for b_title, b_desc in achievements:
        p = tf10_1.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = COLOR_TEXT_WHITE
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = RGBColor(203, 213, 225)

    # Right Dark Card: Future Scope
    card10_2 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    card10_2.fill.solid()
    card10_2.fill.fore_color.rgb = RGBColor(30, 41, 59)
    card10_2.line.color.rgb = RGBColor(51, 65, 85)

    tb10_2 = s10.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.4))
    tf10_2 = tb10_2.text_frame
    tf10_2.word_wrap = True
    p_head2 = tf10_2.paragraphs[0]
    p_head2.text = "🔮 Future Enhancements"
    p_head2.font.size = Pt(16)
    p_head2.font.bold = True
    p_head2.font.color.rgb = COLOR_SECONDARY
    p_head2.space_after = Pt(12)

    future_work = [
        ("Deep Learning Features: ", "Integrate SuperPoint + SuperGlue for ultra-low texture scenes (fog, dark night skies)."),
        ("Spherical 360° Panoramas: ", "Extend projection engine to dual-axis pan/tilt for full equirectangular spherical panoramas."),
        ("Dynamic Seam Optimization: ", "Implement Graph Cut seam-finding (Boykov-Kolmogorov) to eliminate moving object ghosts."),
        ("Exposure Compensation: ", "Add gain compensation and radiometric calibration across varying camera exposures.")
    ]
    for b_title, b_desc in future_work:
        p = tf10_2.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        r1.text = b_title
        r1.font.bold = True
        r1.font.size = Pt(12)
        r1.font.color.rgb = COLOR_TEXT_WHITE
        r2 = p.add_run()
        r2.text = b_desc
        r2.font.size = Pt(12)
        r2.font.color.rgb = RGBColor(203, 213, 225)

    # Save presentation
    prs.save(output_path)
    print(f"✅ Successfully generated 10-slide presentation at: {output_path}")

if __name__ == "__main__":
    create_deck("/Users/priyam/Desktop/CrimeScenceDetecion/Panorama_Stitching_Presentation.pptx")


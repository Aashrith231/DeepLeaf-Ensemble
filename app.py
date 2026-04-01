"""
Intelligent Deforestation & Land Degradation Monitoring System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Computer Vision and Image Processing based analysis of satellite
/ aerial imagery for vegetation health, land cover classification,
and deforestation detection.
"""

import io
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from utils.vegetation import (
    compute_exg,
    compute_grvi,
    compute_pseudo_ndvi,
    compute_vari,
    vegetation_mask,
    vegetation_health_score,
    create_vegetation_heatmap,
    create_ndvi_colormap,
    create_index_visualization,
)
from utils.segmentation import (
    segment_land_cover,
    classify_clusters,
    create_classified_map,
    aggregate_class_areas,
    create_overlay,
    LAND_CLASSES,
)
from utils.change_detection import (
    detect_changes,
    detect_deforestation,
    create_change_overlay,
    compute_change_statistics,
    create_diff_heatmap,
)

# ──────────────────────────── Page Config ─────────────────────────────

st.set_page_config(
    page_title="Deforestation Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ──────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f4c2e 0%, #1a6b3c 40%, #2d8f4e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(15, 76, 46, 0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.88;
        margin: 0;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);
        border: 1px solid #d4edda;
        padding: 1.4rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card h3 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #1a6b3c;
    }
    .metric-card p {
        font-size: 0.85rem;
        color: #555;
        margin: 0.3rem 0 0 0;
        font-weight: 500;
    }

    .metric-bad h3  { color: #c0392b; }
    .metric-warn h3 { color: #e67e22; }
    .metric-ok h3   { color: #27ae60; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a6b3c;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #d4edda;
    }

    .info-box {
        background: #f0faf4;
        border-left: 4px solid #1a6b3c;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #333;
    }

    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 1rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        border: 1px solid #ccc;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f4c2e 0%, #1a5c36 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #e8f5e9 !important;
    }
    div[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1a6b3c, #2d8f4e);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(26,107,60,0.4);
    }

    .about-card {
        background: #f8fffe;
        border: 1px solid #d4edda;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────── Helpers ──────────────────────────────────

def load_uploaded_image(uploaded_file):
    """Convert a Streamlit UploadedFile to a BGR numpy array."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def score_color_class(score):
    if score >= 60:
        return "metric-ok"
    if score >= 35:
        return "metric-warn"
    return "metric-bad"


def score_label(score):
    if score >= 70:
        return "Healthy"
    if score >= 50:
        return "Moderate"
    if score >= 30:
        return "Degraded"
    return "Critical"


# ─────────────────────────── Sidebar ──────────────────────────────────

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        [
            "Home",
            "Vegetation Analysis",
            "Land Cover Classification",
            "Deforestation Detection",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='opacity:0.6'>Deforestation Monitor v1.0<br>"
        "Computer Vision & Image Processing</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════ HOME PAGE ════════════════════════════════

if page == "Home":
    st.markdown(
        """
    <div class="main-header">
        <h1>Intelligent Deforestation &amp; Land Degradation Monitoring</h1>
        <p>Computer Vision and Image Processing &mdash; Satellite / Aerial Image Analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
        Upload satellite or aerial images to analyse vegetation health,
        classify land cover types, and detect deforestation between two
        time-period images &mdash; all powered by classical computer-vision
        algorithms running locally in your browser.
    </div>
    """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    features = [
        (
            "Vegetation Analysis",
            "Compute vegetation indices (ExG, GRVI, pseudo-NDVI, VARI) and generate health heatmaps to assess vegetation density and vigour.",
        ),
        (
            "Land Cover Classification",
            "Segment satellite imagery into land-cover classes (forest, water, urban, barren, agriculture) using K-Means clustering in a multi-colour-space feature set.",
        ),
        (
            "Deforestation Detection",
            "Compare two time-period images to detect areas of vegetation loss (deforestation) and gain (regrowth) with quantitative change statistics.",
        ),
    ]

    for col, (title, desc) in zip(cols, features):
        with col:
            st.markdown(f'<div class="about-card"><h4 style="color:#1a6b3c">{title}</h4><p style="font-size:0.9rem">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-title">How It Works</p>', unsafe_allow_html=True)

    st.markdown(
        """
    1. **Upload** a satellite or aerial photograph (JPEG / PNG).
    2. The system computes **vegetation indices** from the RGB channels.
    3. **K-Means clustering** segments the image into land-cover regions.
    4. For change detection, upload a *before* and *after* image pair.
    5. Results are displayed as interactive charts, heatmaps, and overlays.
    """
    )

    st.markdown('<p class="section-title">Methodology Overview</p>', unsafe_allow_html=True)

    method_cols = st.columns(2)
    with method_cols[0]:
        st.markdown(
            """
        **Vegetation Indices (RGB-based)**
        | Index | Formula |
        |-------|---------|
        | Excess Green (ExG) | `2g − r − b` (normalised) |
        | GRVI | `(G − R) / (G + R)` |
        | Pseudo-NDVI | `(G − R) / (G + R)` (NIR ≈ G) |
        | VARI | `(G − R) / (G + R − B)` |
        """
        )

    with method_cols[1]:
        st.markdown(
            """
        **Land Cover Segmentation**
        - Feature vector: BGR + HSV + CIELAB (9-D)
        - K-Means clustering (k = 6)
        - Rule-based class labelling using average
          brightness, greenness & blueness of each cluster

        **Change Detection**
        - Excess-Green differencing between two dates
        - Morphological cleanup → deforestation / regrowth masks
        """
        )


# ═══════════════════════ VEGETATION ANALYSIS ══════════════════════════

elif page == "Vegetation Analysis":
    st.markdown(
        '<div class="main-header"><h1>Vegetation Health Analysis</h1>'
        "<p>Compute vegetation indices and generate health heatmaps</p></div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a satellite / aerial image",
        type=["jpg", "jpeg", "png", "tif", "bmp"],
        key="veg_upload",
    )

    if uploaded is not None:
        img = load_uploaded_image(uploaded)
        if img is None:
            st.error("Could not decode image. Please upload a valid file.")
        else:
            with st.spinner("Analysing vegetation..."):
                score, veg_frac, avg_green = vegetation_health_score(img)
                heatmap = create_vegetation_heatmap(img)
                ndvi_map, ndvi_raw = create_ndvi_colormap(img)
                exg = compute_exg(img)
                grvi = compute_grvi(img)
                vari = compute_vari(img)
                mask = vegetation_mask(img)

            # ── Metrics row ──
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                cls = score_color_class(score)
                st.markdown(
                    f'<div class="metric-card {cls}"><h3>{score:.1f}</h3>'
                    f"<p>Health Score ({score_label(score)})</p></div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><h3>{veg_frac*100:.1f}%</h3>'
                    "<p>Vegetation Cover</p></div>",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><h3>{avg_green:.3f}</h3>'
                    "<p>Avg Greenness (ExG)</p></div>",
                    unsafe_allow_html=True,
                )
            with m4:
                non_veg = (1 - veg_frac) * 100
                cls_nv = "metric-bad" if non_veg > 60 else ("metric-warn" if non_veg > 40 else "metric-ok")
                st.markdown(
                    f'<div class="metric-card {cls_nv}"><h3>{non_veg:.1f}%</h3>'
                    "<p>Non-Vegetated Area</p></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # ── Images row ──
            st.markdown('<p class="section-title">Visual Results</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(bgr_to_rgb(img), caption="Original Image", use_container_width=True)
            with c2:
                st.image(bgr_to_rgb(heatmap), caption="Vegetation Heatmap (ExG)", use_container_width=True)
            with c3:
                st.image(bgr_to_rgb(ndvi_map), caption="Pseudo-NDVI Map", use_container_width=True)

            c4, c5, c6 = st.columns(3)
            with c4:
                st.image(mask, caption="Vegetation Mask", use_container_width=True, clamp=True)
            with c5:
                grvi_vis = create_index_visualization(grvi, cv2.COLORMAP_VIRIDIS)
                st.image(bgr_to_rgb(grvi_vis), caption="GRVI Map", use_container_width=True)
            with c6:
                vari_vis = create_index_visualization(vari, cv2.COLORMAP_PLASMA)
                st.image(bgr_to_rgb(vari_vis), caption="VARI Map", use_container_width=True)

            # ── Histogram ──
            st.markdown('<p class="section-title">Index Distributions</p>', unsafe_allow_html=True)
            fig = go.Figure()
            for name, data, color in [
                ("ExG", exg.ravel(), "#27ae60"),
                ("GRVI", grvi.ravel(), "#2980b9"),
                ("VARI", np.clip(vari.ravel(), -2, 2), "#8e44ad"),
            ]:
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=name,
                        opacity=0.55,
                        marker_color=color,
                        nbinsx=120,
                    )
                )
            fig.update_layout(
                barmode="overlay",
                template="plotly_white",
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(orientation="h", y=1.08),
                xaxis_title="Index Value",
                yaxis_title="Pixel Count",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════ LAND COVER CLASSIFICATION ════════════════════════

elif page == "Land Cover Classification":
    st.markdown(
        '<div class="main-header"><h1>Land Cover Classification</h1>'
        "<p>K-Means segmentation &amp; rule-based land-type labelling</p></div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a satellite / aerial image",
        type=["jpg", "jpeg", "png", "tif", "bmp"],
        key="seg_upload",
    )

    n_clusters = st.slider("Number of clusters (K)", 3, 10, 6, key="seg_k")

    if uploaded is not None:
        img = load_uploaded_image(uploaded)
        if img is None:
            st.error("Could not decode image.")
        else:
            with st.spinner("Segmenting land cover (this may take a few seconds)..."):
                labels, kmeans = segment_land_cover(img, n_clusters=n_clusters)
                cluster_info = classify_clusters(img, labels)
                classified_map = create_classified_map(labels, cluster_info)
                overlay_img = create_overlay(img, classified_map)
                class_areas = aggregate_class_areas(cluster_info)

            # ── Legend ──
            legend_html = ""
            for cls, props in LAND_CLASSES.items():
                if cls in class_areas:
                    r, g, b = props["rgb"]
                    legend_html += (
                        f'<span class="legend-item">'
                        f'<span class="legend-dot" style="background:rgb({r},{g},{b})"></span>'
                        f"{cls} ({class_areas[cls]:.1f}%)</span>"
                    )
            st.markdown(legend_html, unsafe_allow_html=True)

            # ── Images ──
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(bgr_to_rgb(img), caption="Original", use_container_width=True)
            with c2:
                st.image(bgr_to_rgb(classified_map), caption="Classified Map", use_container_width=True)
            with c3:
                st.image(bgr_to_rgb(overlay_img), caption="Overlay", use_container_width=True)

            # ── Pie chart ──
            st.markdown('<p class="section-title">Land Cover Distribution</p>', unsafe_allow_html=True)

            pie_colors = [
                f"rgb{LAND_CLASSES[cls]['rgb']}" for cls in class_areas.keys()
            ]
            fig = go.Figure(
                go.Pie(
                    labels=list(class_areas.keys()),
                    values=list(class_areas.values()),
                    hole=0.45,
                    marker=dict(colors=pie_colors),
                    textinfo="label+percent",
                    textfont_size=13,
                )
            )
            fig.update_layout(
                template="plotly_white",
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Class-wise metrics ──
            st.markdown('<p class="section-title">Class-wise Summary</p>', unsafe_allow_html=True)
            metric_cols = st.columns(len(class_areas))
            for col, (cls, area) in zip(metric_cols, class_areas.items()):
                with col:
                    st.markdown(
                        f'<div class="metric-card"><h3>{area:.1f}%</h3>'
                        f"<p>{cls}</p></div>",
                        unsafe_allow_html=True,
                    )


# ═══════════════════ DEFORESTATION DETECTION ══════════════════════════

elif page == "Deforestation Detection":
    st.markdown(
        '<div class="main-header"><h1>Deforestation Detection</h1>'
        "<p>Compare two time-period images to detect vegetation loss &amp; gain</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        "Upload two images of the <strong>same area</strong> taken at different times. "
        "The system will detect areas where vegetation was lost (deforestation, shown in "
        "<span style='color:red;font-weight:600'>red</span>) or gained (regrowth, shown in "
        "<span style='color:green;font-weight:600'>green</span>)."
        "</div>",
        unsafe_allow_html=True,
    )

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        before_file = st.file_uploader(
            "Before (earlier date)", type=["jpg", "jpeg", "png", "tif", "bmp"], key="cd_before"
        )
    with col_up2:
        after_file = st.file_uploader(
            "After (later date)", type=["jpg", "jpeg", "png", "tif", "bmp"], key="cd_after"
        )

    if before_file and after_file:
        img_before = load_uploaded_image(before_file)
        img_after = load_uploaded_image(after_file)

        if img_before is None or img_after is None:
            st.error("Could not decode one of the images.")
        else:
            with st.spinner("Detecting deforestation..."):
                before_r, after_r, defor_mask, regr_mask = detect_deforestation(
                    img_before, img_after
                )
                overlay = create_change_overlay(after_r, defor_mask, regr_mask)
                stats = compute_change_statistics(defor_mask, regr_mask)
                diff_heat = create_diff_heatmap(img_before, img_after)

            # ── Metrics ──
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                cls = "metric-bad" if stats["deforested_pct"] > 10 else "metric-warn"
                st.markdown(
                    f'<div class="metric-card {cls}"><h3>{stats["deforested_pct"]:.1f}%</h3>'
                    "<p>Deforested Area</p></div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card metric-ok"><h3>{stats["regrown_pct"]:.1f}%</h3>'
                    "<p>Regrowth Area</p></div>",
                    unsafe_allow_html=True,
                )
            with m3:
                net = stats["net_change_pct"]
                cls_net = "metric-ok" if net >= 0 else "metric-bad"
                sign = "+" if net >= 0 else ""
                st.markdown(
                    f'<div class="metric-card {cls_net}"><h3>{sign}{net:.1f}%</h3>'
                    "<p>Net Vegetation Change</p></div>",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><h3>{stats["unchanged_pct"]:.1f}%</h3>'
                    "<p>Unchanged Area</p></div>",
                    unsafe_allow_html=True,
                )

            # ── Images ──
            st.markdown('<p class="section-title">Comparison</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(bgr_to_rgb(before_r), caption="Before", use_container_width=True)
            with c2:
                st.image(bgr_to_rgb(after_r), caption="After", use_container_width=True)
            with c3:
                st.image(bgr_to_rgb(overlay), caption="Detected Changes", use_container_width=True)

            c4, c5 = st.columns(2)
            with c4:
                st.image(defor_mask, caption="Deforestation Mask", use_container_width=True, clamp=True)
            with c5:
                st.image(bgr_to_rgb(diff_heat), caption="Change Intensity Heatmap", use_container_width=True)

            # ── Bar chart ──
            st.markdown('<p class="section-title">Change Statistics</p>', unsafe_allow_html=True)
            fig = go.Figure(
                go.Bar(
                    x=["Deforested", "Regrowth", "Unchanged"],
                    y=[stats["deforested_pct"], stats["regrown_pct"], stats["unchanged_pct"]],
                    marker_color=["#e74c3c", "#27ae60", "#95a5a6"],
                    text=[
                        f'{stats["deforested_pct"]:.1f}%',
                        f'{stats["regrown_pct"]:.1f}%',
                        f'{stats["unchanged_pct"]:.1f}%',
                    ],
                    textposition="outside",
                )
            )
            fig.update_layout(
                template="plotly_white",
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
                yaxis_title="Area (%)",
                yaxis_range=[0, 110],
            )
            st.plotly_chart(fig, use_container_width=True)



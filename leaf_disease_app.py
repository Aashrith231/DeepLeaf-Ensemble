"""
Plant Leaf Disease Classifier (Frontend)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upload a leaf image and get disease prediction using a trained Keras model.

This is a separate Streamlit app to avoid changing the existing `app.py`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image


# ──────────────────────────── Page Config ─────────────────────────────

st.set_page_config(
    page_title="Plant disease Identification",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────── Styling (non-AI look) ───────────────────

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero {
    background: radial-gradient(1200px 500px at 10% 10%, rgba(46, 204, 113, 0.22), transparent 60%),
                radial-gradient(900px 400px at 95% 30%, rgba(52, 152, 219, 0.16), transparent 55%),
                linear-gradient(135deg, #0e3b2b 0%, #0b5137 40%, #0a6b44 100%);
    padding: 2.4rem 2.0rem;
    border-radius: 18px;
    color: white;
    box-shadow: 0 10px 40px rgba(0,0,0,0.20);
    margin-bottom: 1.2rem;
  }
  .hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 0.35rem 0; letter-spacing: -0.6px; }
  .hero p { font-size: 1.03rem; opacity: 0.9; margin: 0; font-weight: 350; max-width: 65ch; }

  .card {
    background: linear-gradient(135deg, #ffffff 0%, #f7fffb 100%);
    border: 1px solid rgba(46, 204, 113, 0.25);
    border-radius: 16px;
    padding: 1.2rem 1.2rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
  }
  .muted { color: rgba(255,255,255,0.85); }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 0.35rem 0.75rem;
    border: 1px solid rgba(0,0,0,0.10);
    background: rgba(255,255,255,0.92);
    font-size: 0.9rem;
    font-weight: 600;
  }
  .pill b { font-weight: 800; }

  .ok { border-color: rgba(39, 174, 96, 0.28); }
  .warn { border-color: rgba(230, 126, 34, 0.30); }

  .stButton > button {
    background: linear-gradient(135deg, #0a6b44, #27ae60);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    transition: all 0.18s ease;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(39, 174, 96, 0.25);
  }

  footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────── Data helpers ────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    class_names: List[str]
    img_size: Tuple[int, int] = (224, 224)


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def resolve_class_names(dataset_dir: Optional[str], class_names_json: Optional[str]) -> List[str]:
    """
    Priority:
    1) class_names.json (recommended; guarantees correct order)
    2) dataset directory subfolders sorted alphabetically (matches image_dataset_from_directory default)
    """
    if class_names_json:
        obj = _safe_read_json(class_names_json)
        if isinstance(obj, dict) and isinstance(obj.get("class_names"), list):
            return [str(x) for x in obj["class_names"]]
        if isinstance(obj, list):
            return [str(x) for x in obj]

    if dataset_dir and os.path.isdir(dataset_dir):
        classes = [
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
        ]
        classes.sort()
        return classes

    return []


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str) -> tf.keras.Model:
    # Prefer compile=False for faster, more robust loading.
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(img: Image.Image, img_size: Tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(img_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def top_k_predictions(
    probs: np.ndarray, class_names: List[str], k: int = 3
) -> List[Tuple[str, float]]:
    probs = probs.flatten()
    idxs = np.argsort(probs)[::-1][:k]
    out: List[Tuple[str, float]] = []
    for i in idxs:
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        out.append((name, float(probs[i])))
    return out


def confidence_label(p: float) -> Tuple[str, str]:
    if p >= 0.85:
        return "High confidence", "ok"
    if p >= 0.60:
        return "Moderate confidence", "warn"
    return "Low confidence", "warn"


# ──────────────────────────── Sidebar ─────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")
    st.caption("Point this app to your trained model and class list.")

    model_path = st.text_input(
        "Model file (.h5)",
        value=os.environ.get("LEAF_MODEL_PATH", ""),
        placeholder=r"C:\path\to\MobileNetV2.h5",
    )

    class_names_json = st.text_input(
        "Optional: class_names.json",
        value=os.environ.get("LEAF_CLASSNAMES_JSON", ""),
        placeholder=r"C:\path\to\class_names.json",
    )

    dataset_dir = st.text_input(
        "Optional: dataset folder (for class order)",
        value=os.environ.get("LEAF_DATASET_DIR", ""),
        placeholder=r"C:\path\to\Plant_leave_diseases_dataset_with_augmentation",
    )

    k_top = st.slider("Top-k results", 1, 5, 3)
    st.markdown("---")
    st.caption("Tip: Save `class_names.json` once during training for perfect label order.")


# ──────────────────────────── Main UI ─────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>Leaf Health Check</h1>
  <p>
    Upload a leaf image to identify the most likely disease class from your trained CNN model.
    Designed for a clean demo experience (no chat, no AI jargon).
  </p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload a leaf photo")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.caption("Best results: centered leaf, good lighting, minimal background clutter.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model status")
    class_names = resolve_class_names(dataset_dir or None, class_names_json or None)

    if not model_path:
        st.info("Add your `.h5` path in the sidebar to enable predictions.")
    elif not os.path.isfile(model_path):
        st.error("Model path not found. Check the sidebar path.")
    else:
        st.success("Model path looks valid.")

    if class_names:
        st.write(f"Classes detected: **{len(class_names)}**")
        with st.expander("Show class names"):
            st.write(class_names)
    else:
        st.warning(
            "Class names not set. Provide `class_names.json` or a dataset folder to map predictions to labels."
        )

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("")
st.markdown("### Prediction")

col_a, col_b = st.columns([0.55, 0.45], gap="large")

with col_a:
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
    else:
        st.info("Upload an image to preview it here.")

with col_b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.button("Run prediction", use_container_width=True, disabled=(uploaded is None or not model_path)):
        if not os.path.isfile(model_path):
            st.error("Model file not found. Fix the path in sidebar.")
        elif uploaded is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Running inference..."):
                model = load_keras_model(model_path)
                img = Image.open(uploaded)
                x = preprocess_image(img, (224, 224))
                probs = model.predict(x, verbose=0)
                preds = top_k_predictions(probs, class_names, k=k_top) if class_names else [
                    (f"class_{i}", float(p)) for i, p in enumerate(probs.flatten())
                ][:k_top]

            best_name, best_p = preds[0]
            conf_text, conf_cls = confidence_label(best_p)

            st.markdown(
                f'<span class="pill {conf_cls}">Prediction: <b>{best_name}</b> · {best_p:.2%}</span>',
                unsafe_allow_html=True,
            )
            st.caption(conf_text)

            df = px.data.tips()  # placeholder to create structure without pandas requirement
            # Build a small dataframe-like dict for plotly without importing pandas
            names = [n for n, _ in preds]
            scores = [p for _, p in preds]
            fig = px.bar(
                x=scores,
                y=names,
                orientation="h",
                text=[f"{p:.2%}" for p in scores],
                labels={"x": "Probability", "y": "Class"},
                title="Top predictions",
            )
            fig.update_traces(marker_color="#27ae60")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Details"):
                st.write(
                    {
                        "model_path": model_path,
                        "image_size": (224, 224),
                        "top_k": k_top,
                    }
                )

    else:
        st.caption("Click “Run prediction” after uploading an image.")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.caption(
    "Demo UI for midterm evaluation. For best label order, export `class_names.json` during training and point the sidebar to it."
)


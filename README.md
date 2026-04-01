# Intelligent Deforestation & Land Degradation Monitoring

> Computer Vision and Image Processing based system for analysing satellite/aerial
> imagery to monitor vegetation health, classify land cover, and detect deforestation.

---

## Features

| Module | Description |
|--------|-------------|
| **Vegetation Analysis** | Computes Excess Green Index (ExG), GRVI, pseudo-NDVI and VARI from RGB imagery; generates colour-coded heatmaps and a 0–100 health score |
| **Land Cover Classification** | K-Means clustering on a 9-D feature vector (BGR + HSV + CIELAB) with rule-based labelling into Dense Forest, Sparse Vegetation, Water, Urban, Barren, and Agricultural classes |
| **Deforestation Detection** | Temporal differencing of the ExG index between two images to detect vegetation loss (deforestation) and gain (regrowth), with interactive change statistics |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py
```

The web app opens at **http://localhost:8501**.

---

## Leaf Disease Classifier (Frontend)

This repo also includes a separate Streamlit UI for your plant leaf disease model:

```bash
streamlit run leaf_disease_app.py
```

In the sidebar, set:
- **Model file (.h5)**: path to your trained Keras model (e.g. `MobileNetV2.h5`)
- **Optional class_names.json**: list of class names in the exact training order
- **Optional dataset folder**: fallback to infer class order from subfolders (alphabetical)

---

## Project Structure

```
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── utils/
│   ├── __init__.py
│   ├── vegetation.py          # Vegetation index computation & heatmaps
│   ├── segmentation.py        # K-Means land cover segmentation
│   └── change_detection.py    # Temporal change / deforestation detection
└── sample_images/             # Sample satellite images for demo
```

---

## Technical Methodology

### Vegetation Indices (RGB-based)

| Index | Formula | Purpose |
|-------|---------|---------|
| Excess Green (ExG) | `2g − r − b` (normalised) | Emphasise green vegetation |
| GRVI | `(G − R) / (G + R)` | Green-to-red reflectance ratio |
| Pseudo-NDVI | `(G − R) / (G + R)` | Approximate NDVI (NIR ≈ Green) |
| VARI | `(G − R) / (G + R − B)` | Atmospherically resistant index |

### Land Cover Segmentation

1. Convert image to BGR, HSV, and CIELAB colour spaces.
2. Concatenate into a 9-dimensional feature vector per pixel.
3. Apply K-Means clustering (default k = 6).
4. Label each cluster using rule-based classification (brightness, greenness, blueness).

### Deforestation Detection

1. Compute ExG for both *before* and *after* images.
2. Difference the indices; threshold to create deforestation and regrowth masks.
3. Apply morphological cleanup (closing + opening).
4. Generate overlay visualisation and change statistics.

---

## Technologies

- **Python 3.10+**
- **OpenCV** – image processing
- **NumPy** – numerical computation
- **scikit-learn** – K-Means clustering
- **Streamlit** – interactive web UI
- **Plotly** – data visualisation


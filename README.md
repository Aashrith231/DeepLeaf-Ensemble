# DeepLeaf-Ensemble (Phase 1 Baselines + Demo + XAI)

Deep learning project for **plant leaf disease identification** using **transfer learning** and a controlled **comparative study** of pretrained CNN backbones. Includes a **Streamlit** demo app and **Grad-CAM** tooling. The broader final-project direction is **model ensembling** (planned), built on these baselines.

---

## What’s included

- **Baseline training & evaluation** (TensorFlow/Keras): MobileNetV2, ResNet50, VGG16, EfficientNetB0, DenseNet121
- **Metrics**: val accuracy, weighted/macro precision/recall/F1, balanced accuracy, MCC, training time
- **Results export**: `comparison_results.csv`
- **Research-style report**: `RESEARCH_REPORT.md` + `RESEARCH_REPORT.pdf`
- **Frontend demo**: upload leaf image → top‑k predictions + confidence chart (`leaf_disease_app.py`)
- **Grad-CAM** (saved weights compatible): `gradcam_saved_weights.py`

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the leaf disease frontend
streamlit run leaf_disease_app.py
```

The web app opens at **http://localhost:8501**.

---

## Frontend: Leaf Disease Classifier

In the sidebar, set:
- **Model file (.h5)**: path to your trained Keras model (example: `MobileNetV2.h5`)
- **class_names.json**: class names in the exact training order (recommended)
- **dataset folder** (optional): fallback to infer class order from subfolders (alphabetical)

Tip: avoid committing the dataset/model to GitHub. This repo ships the code; you keep the dataset/model locally or in Drive.

### Create `class_names.json`
After loading the dataset in Colab (so `class_names` exists), run:

```python
import json, os

MODEL_SAVE_DIR = "/content/drive/MyDrive/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

with open(f"{MODEL_SAVE_DIR}/class_names.json", "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names}, f, indent=2)
```

---

## Grad-CAM heatmap (from saved weights)

This script works even when `load_model()` fails on newer Keras (e.g. legacy `TrueDivide`), by rebuilding the same architecture and using `load_weights()`.

```bash
python gradcam_saved_weights.py \
  --weights "path/to/your/MobileNetV2.h5" \
  --image "path/to/leaf.jpg" \
  --class_names_json "class_names.json" \
  --arch MobileNetV2 \
  --out heatmap.png
```

Supported `--arch` values: `MobileNetV2`, `ResNet50`, `VGG16`, `EfficientNetB0`, `DenseNet121`.

---

## Project Structure

```
├── RESEARCH_REPORT.md                     # research paper style report (markdown)
├── RESEARCH_REPORT.pdf                    # exported PDF report
├── comparison_results.csv                 # metrics export (from your run)
├── export_research_pdf.py                 # regenerate PDF from markdown
├── leaf_disease_app.py        # Streamlit: upload → predict (plant disease)
├── gradcam_saved_weights.py   # Grad-CAM from saved weights (.h5)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── Comparative_Study_CNN_refactored.ipynb  # comparative study (presentation-friendly)
├── Comparative_Study_CNN.ipynb             # original comparative notebook
├── XAI_Heatmap_from_SavedModel.ipynb       # XAI occlusion heatmap (saved model)
├── XAI_Heatmap_ImageNet_MobileNetV2.ipynb  # XAI heatmap demo (ImageNet)
├── class_names.json            # (optional) label mapping for frontend
└── utils/                      # extra utilities (kept from earlier work)
```

---

## Method summary (midterm)

- **Dataset loading**: `tf.keras.utils.image_dataset_from_directory` (folder-per-class), resized to **224×224**
- **Transfer learning**: ImageNet pretrained backbone, `include_top=False`, frozen base
- **Common head**: GAP → Dense(256, ReLU) → Dropout(0.5) → Softmax
- **Models**: MobileNetV2, ResNet50, VGG16, EfficientNetB0, DenseNet121
- **Metrics**: validation accuracy, precision/recall/F1 (macro & weighted), balanced accuracy, MCC, training time

---

## Results (from the recorded run)

See `comparison_results.csv`. In the captured run, **ResNet50** ranks best on validation accuracy, macro F1, balanced accuracy, and MCC.

---

## Technologies

- **Python 3.10+**
- **TensorFlow/Keras** – model training & inference
- **Streamlit** – demo frontend
- **NumPy / OpenCV / Matplotlib** – preprocessing & visualization
- **scikit-learn** – evaluation metrics


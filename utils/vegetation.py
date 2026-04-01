import cv2
import numpy as np


def compute_exg(image):
    """Excess Green Index: 2g - r - b (normalized channels)."""
    img = image.astype(np.float64)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    total = R + G + B + 1e-6
    r, g, b = R / total, G / total, B / total
    return 2 * g - r - b


def compute_grvi(image):
    """Green-Red Vegetation Index: (G - R) / (G + R)."""
    img = image.astype(np.float64)
    G, R = img[:, :, 1], img[:, :, 2]
    return (G - R) / (G + R + 1e-6)


def compute_pseudo_ndvi(image):
    """
    Approximate NDVI using RGB channels.
    Real NDVI = (NIR - Red) / (NIR + Red).
    We approximate NIR using the Green channel since vegetation
    reflects strongly in both NIR and green bands.
    """
    img = image.astype(np.float64)
    R, G = img[:, :, 2], img[:, :, 1]
    nir_approx = G
    return (nir_approx - R) / (nir_approx + R + 1e-6)


def compute_vari(image):
    """Visible Atmospherically Resistant Index: (G - R) / (G + R - B)."""
    img = image.astype(np.float64)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (G - R) / (G + R - B + 1e-6)


def vegetation_mask(image, threshold=0.05):
    """Binary mask of vegetation regions using Excess Green Index."""
    exg = compute_exg(image)
    mask = (exg > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def vegetation_health_score(image):
    """
    Returns (score, vegetation_fraction, avg_greenness).
    Score is 0-100 representing overall vegetation health.
    """
    exg = compute_exg(image)
    mask = vegetation_mask(image)
    veg_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    veg_fraction = veg_pixels / total_pixels

    if veg_pixels > 0:
        avg_greenness = np.mean(exg[mask > 0])
    else:
        avg_greenness = 0.0

    greenness_norm = np.clip((avg_greenness + 0.33) / 0.66, 0, 1)
    score = veg_fraction * 60 + greenness_norm * 40
    score = np.clip(score, 0, 100)

    return float(score), float(veg_fraction), float(avg_greenness)


def create_vegetation_heatmap(image):
    """Color heatmap of vegetation density (blue=low, red=high)."""
    exg = compute_exg(image)
    exg_norm = np.clip((exg - exg.min()) / (exg.max() - exg.min() + 1e-6), 0, 1)
    exg_uint8 = (exg_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(exg_uint8, cv2.COLORMAP_JET)
    return heatmap


def create_ndvi_colormap(image):
    """Color-coded pseudo-NDVI map (red=bare, yellow=sparse, green=dense)."""
    ndvi = compute_pseudo_ndvi(image)
    ndvi_norm = np.clip((ndvi + 1) / 2, 0, 1)
    ndvi_uint8 = (ndvi_norm * 255).astype(np.uint8)

    colormap = cv2.applyColorMap(ndvi_uint8, cv2.COLORMAP_HOT)
    return colormap, ndvi


def create_index_visualization(index_array, colormap=cv2.COLORMAP_JET):
    """Generic visualization for any vegetation index array."""
    norm = np.clip(
        (index_array - index_array.min()) / (index_array.max() - index_array.min() + 1e-6),
        0, 1,
    )
    uint8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(uint8, colormap)

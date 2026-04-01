import cv2
import numpy as np

from .vegetation import compute_exg


def _match_sizes(img_a, img_b):
    """Resize both images to a common resolution."""
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    return cv2.resize(img_a, (w, h)), cv2.resize(img_b, (w, h))


def detect_changes(img_before, img_after, threshold=30):
    """
    General change detection between two images.
    Returns resized pair, binary change mask, and raw difference.
    """
    before, after = _match_sizes(img_before, img_after)

    gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_b, gray_a)
    _, change_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

    return before, after, change_mask, diff


def detect_deforestation(img_before, img_after):
    """
    Detect vegetation loss (deforestation) and vegetation gain (regrowth)
    by comparing greenness between two time-period images.
    """
    before, after = _match_sizes(img_before, img_after)

    exg_before = compute_exg(before)
    exg_after = compute_exg(after)

    green_loss = exg_before - exg_after

    deforestation_mask = (green_loss > 0.06).astype(np.uint8) * 255
    regrowth_mask = (green_loss < -0.06).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    deforestation_mask = cv2.morphologyEx(deforestation_mask, cv2.MORPH_CLOSE, kernel)
    deforestation_mask = cv2.morphologyEx(deforestation_mask, cv2.MORPH_OPEN, kernel)
    regrowth_mask = cv2.morphologyEx(regrowth_mask, cv2.MORPH_CLOSE, kernel)
    regrowth_mask = cv2.morphologyEx(regrowth_mask, cv2.MORPH_OPEN, kernel)

    return before, after, deforestation_mask, regrowth_mask


def create_change_overlay(image, deforestation_mask, regrowth_mask):
    """
    Overlay deforestation (red) and regrowth (green) on the image.
    """
    overlay = image.copy()
    overlay[deforestation_mask > 0] = [0, 0, 255]
    overlay[regrowth_mask > 0] = [0, 255, 0]
    return cv2.addWeighted(image, 0.55, overlay, 0.45, 0)


def compute_change_statistics(deforestation_mask, regrowth_mask):
    """Return a dictionary of change-detection statistics."""
    total = deforestation_mask.size
    deforested = int(np.sum(deforestation_mask > 0))
    regrown = int(np.sum(regrowth_mask > 0))

    return {
        "total_pixels": total,
        "deforested_pixels": deforested,
        "regrown_pixels": regrown,
        "deforested_pct": round(deforested / total * 100, 2),
        "regrown_pct": round(regrown / total * 100, 2),
        "net_change_pct": round((regrown - deforested) / total * 100, 2),
        "unchanged_pct": round((total - deforested - regrown) / total * 100, 2),
    }


def create_diff_heatmap(img_before, img_after):
    """
    Create a heatmap showing intensity of change between two images.
    """
    before, after = _match_sizes(img_before, img_after)
    gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_a = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float64)

    diff = np.abs(gray_b - gray_a)
    diff_norm = np.clip(diff / (diff.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)

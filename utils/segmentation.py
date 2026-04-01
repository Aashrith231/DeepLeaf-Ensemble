import cv2
import numpy as np
from sklearn.cluster import KMeans


LAND_CLASSES = {
    "Dense Forest":       {"color": (34, 139, 34),   "rgb": (34, 139, 34)},
    "Sparse Vegetation":  {"color": (144, 238, 144), "rgb": (144, 238, 144)},
    "Water Body":         {"color": (30, 80, 200),   "rgb": (200, 80, 30)},
    "Urban / Built-up":   {"color": (150, 150, 150), "rgb": (150, 150, 150)},
    "Barren Land":        {"color": (180, 140, 100), "rgb": (100, 140, 180)},
    "Agricultural Land":  {"color": (200, 200, 60),  "rgb": (60, 200, 200)},
}


def segment_land_cover(image, n_clusters=6):
    """
    Segment image into land cover regions using K-means clustering
    on combined BGR + HSV + LAB feature space.
    """
    small = cv2.resize(image, (300, 300))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)

    features = np.concatenate(
        [
            small.reshape(-1, 3).astype(np.float64),
            hsv.reshape(-1, 3).astype(np.float64),
            lab.reshape(-1, 3).astype(np.float64),
        ],
        axis=1,
    )

    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels_small = kmeans.fit_predict(features).reshape(300, 300)

    labels = cv2.resize(
        labels_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return labels, kmeans


def classify_clusters(image, labels):
    """
    Assign a land cover class to each cluster based on its average
    colour properties (brightness, greenness, blueness).
    """
    n_clusters = labels.max() + 1
    cluster_info = {}

    for i in range(n_clusters):
        mask = labels == i
        if not np.any(mask):
            continue

        pixels = image[mask].astype(np.float64)
        avg_b, avg_g, avg_r = pixels.mean(axis=0)

        brightness = (avg_r + avg_g + avg_b) / 3
        greenness = avg_g - (avg_r + avg_b) / 2
        blueness = avg_b - (avg_r + avg_g) / 2

        if greenness > 15 and avg_g > 90:
            label = "Dense Forest" if brightness < 130 else "Sparse Vegetation"
        elif blueness > 15 and brightness < 110:
            label = "Water Body"
        elif brightness > 155 and abs(avg_r - avg_g) < 25 and abs(avg_g - avg_b) < 25:
            label = "Urban / Built-up"
        elif avg_r > avg_g and avg_r > 110 and greenness < -5:
            label = "Barren Land"
        elif greenness > 0 and brightness > 120:
            label = "Agricultural Land"
        elif brightness < 80:
            label = "Water Body"
        else:
            label = "Barren Land"

        area_pct = np.sum(mask) / mask.size * 100
        cluster_info[i] = {
            "class": label,
            "area_percent": round(area_pct, 2),
            "color_bgr": LAND_CLASSES[label]["color"],
            "color_rgb": LAND_CLASSES[label]["rgb"],
        }

    return cluster_info


def create_classified_map(labels, cluster_info):
    """Produce a colour-coded classified-land-cover image."""
    h, w = labels.shape
    classified = np.zeros((h, w, 3), dtype=np.uint8)

    for cid, info in cluster_info.items():
        classified[labels == cid] = info["color_bgr"]

    return classified


def aggregate_class_areas(cluster_info):
    """Sum area percentages for each land cover class across clusters."""
    areas = {}
    for info in cluster_info.values():
        cls = info["class"]
        areas[cls] = areas.get(cls, 0) + info["area_percent"]
    return areas


def create_overlay(image, classified_map, alpha=0.45):
    """Blend the classified map onto the original image."""
    return cv2.addWeighted(image, 1 - alpha, classified_map, alpha, 0)

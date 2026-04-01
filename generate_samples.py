"""Generate synthetic satellite-like sample images for demo purposes."""

import cv2
import numpy as np
import os

os.makedirs("sample_images", exist_ok=True)
H, W = 512, 512


def _noise(shape, scale=10):
    return np.random.normal(0, scale, shape).astype(np.float64)


def make_forest_image():
    """Create a synthetic aerial image with forest, water, and barren areas."""
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Dense forest (dark green)
    img[:, :] = [30, 90, 25]
    img[:] = np.clip(img.astype(np.float64) + _noise((H, W, 3), 12), 0, 255).astype(np.uint8)

    # Water body (top-right lake)
    cv2.ellipse(img, (400, 80), (100, 60), 0, 0, 360, (140, 60, 20), -1)
    mask_water = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask_water, (400, 80), (100, 60), 0, 0, 360, 255, -1)
    noise = _noise((H, W, 3), 8)
    img[mask_water > 0] = np.clip(
        img[mask_water > 0].astype(np.float64) + noise[mask_water > 0], 0, 255
    ).astype(np.uint8)

    # Sparse vegetation patch (lighter green)
    cv2.rectangle(img, (50, 350), (250, 500), (50, 160, 60), -1)
    region = img[350:500, 50:250].astype(np.float64)
    region += _noise(region.shape, 15)
    img[350:500, 50:250] = np.clip(region, 0, 255).astype(np.uint8)

    # Barren / brown patch
    cv2.circle(img, (300, 350), 70, (60, 120, 160), -1)
    mask_barren = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask_barren, (300, 350), 70, 255, -1)
    img[mask_barren > 0] = np.clip(
        img[mask_barren > 0].astype(np.float64) + _noise((H, W, 3), 10)[mask_barren > 0], 0, 255
    ).astype(np.uint8)

    # Road (gray line)
    cv2.line(img, (0, 250), (512, 280), (130, 130, 130), 6)

    return img


def make_deforested_image(forest_img):
    """Take a forest image and simulate deforestation patches."""
    img = forest_img.copy()

    # Large cleared area (brown/bare soil)
    pts = np.array([[100, 100], [300, 80], [350, 200], [280, 280], [100, 250]])
    cv2.fillPoly(img, [pts], (70, 130, 170))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    img[mask > 0] = np.clip(
        img[mask > 0].astype(np.float64) + _noise((H, W, 3), 15)[mask > 0], 0, 255
    ).astype(np.uint8)

    # Smaller cleared patch
    cv2.circle(img, (420, 350), 55, (80, 140, 180), -1)
    mask2 = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask2, (420, 350), 55, 255, -1)
    img[mask2 > 0] = np.clip(
        img[mask2 > 0].astype(np.float64) + _noise((H, W, 3), 10)[mask2 > 0], 0, 255
    ).astype(np.uint8)

    return img


np.random.seed(42)

forest = make_forest_image()
deforested = make_deforested_image(forest)

cv2.imwrite("sample_images/forest_before.jpg", forest)
cv2.imwrite("sample_images/forest_after.jpg", deforested)

# Urban/mixed scene
urban = np.zeros((H, W, 3), dtype=np.uint8)
urban[:] = [50, 130, 45]  # green background
urban += np.clip(_noise((H, W, 3), 12), -50, 50).astype(np.uint8)

# Buildings (gray rectangles)
for x, y, w2, h2 in [(50, 50, 80, 60), (200, 100, 100, 80), (350, 50, 90, 70)]:
    cv2.rectangle(urban, (x, y), (x + w2, y + h2), (160, 160, 155), -1)

# River
pts_river = np.array([[0, 300], [150, 280], [300, 310], [512, 290], [512, 320], [300, 340], [150, 310], [0, 330]])
cv2.fillPoly(urban, [pts_river], (150, 80, 30))

# Agricultural fields (lighter green rectangles)
cv2.rectangle(urban, (30, 380), (200, 500), (60, 190, 80), -1)
cv2.rectangle(urban, (220, 380), (400, 500), (55, 175, 70), -1)

urban = np.clip(urban.astype(np.float64) + _noise((H, W, 3), 6), 0, 255).astype(np.uint8)
cv2.imwrite("sample_images/mixed_landscape.jpg", urban)

print("Generated sample images in sample_images/")
print("  - forest_before.jpg  (healthy forest)")
print("  - forest_after.jpg   (same area, deforested)")
print("  - mixed_landscape.jpg (urban + forest + water + agriculture)")

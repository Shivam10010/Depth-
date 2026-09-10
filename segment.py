"""
DepthWizard — Stage 3: Segmentation
=====================================
Turns the photo into a simple "map" saying which pixels are
vegetation (green stuff), buildings (gray/white stuff), or ground (everything else).

────────────────── HOW IT WORKS ──────────────────
No machine learning here — just smart color thresholds:

1. Vegetation: In aerial photos, plants look green.
   We check if the GREEN channel is significantly brighter than RED and BLUE.

2. Built-up: If it's NOT vegetation AND it's relatively bright
   (buildings, roads, concrete are light-colored), it's probably built-up.

3. Ground: Everything else (soil, roads, shadows, water).

This is a simple heuristic — it won't be perfect, but it's fast
(no GPU needed) and works surprisingly well for aerial/satellite images.
"""

import os
import numpy as np
import cv2

from config import OUTPUTS_DIR, SEG_COLORS


def segment(image_path, depth_shape=None):
    """Classify every pixel into ground / vegetation / built-up.

    Args:
        image_path:    path to the original photo
        depth_shape:   optional (height, width) — we resize the mask
                       to match the depth map so everything lines up

    Returns:
        labels: 2D array of integers:
            0 = ground
            1 = vegetation
            2 = built-up
        mask_path: where the colored mask PNG was saved
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Can't find image: {image_path}")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Couldn't read image: {image_path}")

    # Resize to match depth map if needed (so everything is the same size)
    if depth_shape is not None:
        target_width = depth_shape[1]
        target_height = depth_shape[0]
        image_bgr = cv2.resize(image_bgr, (target_width, target_height),
                               interpolation=cv2.INTER_AREA)

    # Convert to RGB so we can work with R, G, B channels separately
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # Split into separate color channels (as floats so subtraction works well)
    red = image_rgb[:, :, 2].astype(np.float32)
    green = image_rgb[:, :, 1].astype(np.float32)
    blue = image_rgb[:, :, 0].astype(np.float32)

    # Grayscale version (used for brightness check)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── Vegetation: green is the strongest channel ──────────────────────────
    # We measure "greenness" as: how much greener is it than red AND blue?
    greenness = (green - red) + (green - blue)

    # If greenness > 15, call it vegetation.
    # The number 15 was picked by trial and error for aerial images.
    vegetation_threshold = 15
    is_vegetation = greenness > vegetation_threshold

    # ── Built-up: NOT vegetation AND bright ─────────────────────────────────
    # Buildings, roads, and concrete are light-colored (gray > 100 out of 255).
    brightness_threshold = 100
    is_built_up = (~is_vegetation) & (gray > brightness_threshold)

    # ── Build the label map ─────────────────────────────────────────────────
    # Start with everything = ground (0)
    labels = np.zeros((height, width), dtype=np.uint8)
    labels[is_vegetation] = 1       # green pixels → vegetation
    labels[is_built_up] = 2          # bright non-green → built-up

    # ── Save a pretty visualization ─────────────────────────────────────────
    # Create a color image where each label gets its assigned color
    mask_visual = np.zeros((height, width, 3), dtype=np.uint8)
    for label_id, color_bgr in SEG_COLORS.items():
        mask_visual[labels == label_id] = color_bgr

    mask_path = os.path.join(OUTPUTS_DIR, "segmentation_mask.png")
    cv2.imwrite(mask_path, mask_visual)

    # Print some stats
    veg_pixels = is_vegetation.sum()
    build_pixels = is_built_up.sum()
    ground_pixels = (labels == 0).sum()
    print(f"[segment] Saved mask: {mask_path}")
    print(f"[segment]   Vegetation: {veg_pixels} px ({veg_pixels/labels.size*100:.1f}%)")
    print(f"[segment]   Built-up:   {build_pixels} px ({build_pixels/labels.size*100:.1f}%)")
    print(f"[segment]   Ground:     {ground_pixels} px ({ground_pixels/labels.size*100:.1f}%)")

    return labels, mask_path

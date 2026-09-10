"""
DepthWizard — Stage 2: Depth Estimation
========================================
Given a photo, estimate how far away each pixel is.

Uses Intel's MiDaS model (small version = fast on CPU).
Output: a "depth map" — a grayscale image where bright = close, dark = far.

────────────────── THE BIG IDEA ──────────────────
MiDaS was trained on many photos with known depth. It learned patterns:
  - things at the bottom of the photo tend to be closer (ground plane)
  - edges often mark object boundaries
  - texture density correlates with distance

It doesn't "know" real-world distances (meters) — only relative depth (0 to 1).
We convert that to meters later in Stage 4 (calibration).
"""

import os
import numpy as np
import cv2
import torch

from config import (
    OUTPUTS_DIR, MAX_IMAGE_SIDE,
    MIDAS_REPO, DEPTH_MODEL_NAME, DEPTH_TRANSFORM_NAME,
)


def load_model():
    """Download (once) and load the MiDaS depth model.

    First run downloads ~40 MB of model weights. After that it's cached
    by PyTorch on disk, so subsequent runs are instant.

    Returns:
        model   — the neural network (call it on an image tensor)
        transform — a function that prepares an image for the model
    """
    print(f"[depth] Loading model '{DEPTH_MODEL_NAME}'...")

    # Load the pre-trained model from Intel's GitHub repo via torch.hub
    model = torch.hub.load(MIDAS_REPO, DEPTH_MODEL_NAME, pretrained=True, trust_repo=True)

    # .eval() = evaluation mode (disables dropout, etc. — needed for inference)
    model.eval()
    # Move to CPU so it works on any laptop (no GPU needed)
    model.cpu()

    # The "transforms" module knows how to resize & normalize images for this model
    transforms = torch.hub.load(MIDAS_REPO, "transforms", trust_repo=True)

    # Different MiDaS variants need slightly different preprocessing
    model_name = DEPTH_MODEL_NAME.lower()
    if "small" in model_name:
        transform = transforms.small_transform
    else:
        # dpt_hybrid, dpt_large, etc. all use the DPT transform
        transform = transforms.dpt_transform

    print("[depth] Model loaded! Ready to estimate depth.")
    return model, transform


def _shrink_image(image, max_side):
    """If the image is bigger than max_side pixels on its longest edge,
    shrink it down while keeping the same shape (aspect ratio).

    Why shrink?
      - Bigger images = more pixels = more math = slower.
      - Depth maps don't need full-resolution input.
      - 512px is plenty for a good result.
    """
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image  # already small enough

    # Figure out the scale factor
    scale = max_side / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # INTER_AREA is best for shrinking (less blurry than INTER_LINEAR)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def estimate_depth(image_path, model=None, transform=None):
    """Estimate depth from a photo.

    Pipeline:
        1. Load image (OpenCV reads as BGR, so we convert to RGB)
        2. Shrink to MAX_IMAGE_SIDE
        3. Preprocess with the model's transform
        4. Run the neural network
        5. Normalize output to [0, 1] and save a pretty heatmap

    Args:
        image_path: path to the input photo
        model: pre-loaded model (optional — loads on first call)
        transform: pre-loaded transform (optional)

    Returns:
        depth_normalized — 2D numpy array, values from 0.0 to 1.0
        heatmap_path     — where the pretty PNG was saved
    """
    # Load model if not already loaded
    if model is None or transform is None:
        model, transform = load_model()

    # ── Step 1: Read the image ──────────────────────────────────────────────
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Can't find image: {image_path}")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Couldn't read image (maybe not an image file?): {image_path}")

    # OpenCV reads in BGR order. Neural nets expect RGB.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Step 2: Resize ──────────────────────────────────────────────────────
    image_rgb = _shrink_image(image_rgb, MAX_IMAGE_SIDE)
    original_height, original_width = image_rgb.shape[:2]
    print(f"[depth] Image size: {original_width}x{original_height}")

    # ── Step 3: Preprocess for the model ───────────────────────────────────
    # The transform normalizes pixel values to the range the model expects.
    if "small" in DEPTH_MODEL_NAME.lower():
        # small_transform: give it a numpy array, get a tensor back
        input_tensor = transform(image_rgb)
    else:
        # dpt_transform: expects a dict with "image" key
        input_tensor = transform({"image": image_rgb})
        input_tensor = input_tensor.get("sample", list(input_tensor.values())[0])

    # Add a "batch" dimension: model expects [batch, channels, height, width]
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(torch.device("cpu"))

    # ── Step 4: Run the model ───────────────────────────────────────────────
    # torch.no_grad() = don't track gradients (saves memory, we're not training)
    with torch.no_grad():
        # Forward pass: image → raw depth prediction
        raw_output = model(input_tensor)

        # Resize output back to the original image size
        raw_output = torch.nn.functional.interpolate(
            raw_output.unsqueeze(1),            # add channel dim: [1, 1, h, w]
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()  # remove batch & channel dims → [h, w]

    # Convert PyTorch tensor → numpy array
    depth_raw = raw_output.cpu().numpy()

    # ── Step 5: Normalize to [0, 1] ────────────────────────────────────────
    # The model outputs arbitrary numbers. We scale so min=0, max=1.
    depth_min = depth_raw.min()
    depth_max = depth_raw.max()
    depth_range = depth_max - depth_min

    if depth_range > 1e-8:  # avoid division by zero for flat images
        depth_normalized = (depth_raw - depth_min) / depth_range
    else:
        depth_normalized = np.zeros_like(depth_raw)

    depth_normalized = depth_normalized.astype(np.float32)

    # ── Step 6: Save a pretty heatmap ──────────────────────────────────────
    # Convert 0-1 float → 0-255 uint8, then apply a color map
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    heatmap_path = os.path.join(OUTPUTS_DIR, "depth_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"[depth] Heatmap saved: {heatmap_path}")

    return depth_normalized, heatmap_path

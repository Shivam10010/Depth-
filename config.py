"""
DepthWizard — Config
====================
All the numbers you might want to tweak in one place.

Think of this as the "settings panel" for the whole app.
"""

import os

# ── Where are we? ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)  # create folder if it doesn't exist

# ── Image preprocessing ───────────────────────────────────────────────────────
# Very large images run slowly, so we shrink them.
# MAX_IMAGE_SIDE = the largest edge (width or height) in pixels.
MAX_IMAGE_SIDE = 512  # 512px is enough for a good depth map & runs fast

# ── Depth estimation model ────────────────────────────────────────────────────
# "MiDaS_small"  → small & fast (~1 MB, ~0.2s on CPU). Good default.
# "dpt_hybrid"   → bigger & better (~500 MB, ~2s on CPU).
# Swap this line to try a different model.
DEPTH_MODEL_NAME = "MiDaS_small"
DEPTH_TRANSFORM_NAME = "DPT_Hybrid_MiDaS"
MIDAS_REPO = "intel-isl/MiDaS"

# ── Segmentation (color-based, no ML) ─────────────────────────────────────────
# We label each pixel as 0=ground, 1=vegetation, 2=built-up.
SEG_LABELS = {0: "ground", 1: "vegetation", 2: "built-up"}
# Colors for drawing the mask (BGR order, used by OpenCV)
SEG_COLORS = {0: (120, 120, 120), 1: (34, 139, 34), 2: (180, 130, 100)}

# ── Height calibration ────────────────────────────────────────────────────────
# When there's no real-world elevation data (DEM), we just scale the
# depth map so that the tallest object is this many meters.
DEFAULT_MAX_HEIGHT_M = 50

# ── 3D Mesh settings ──────────────────────────────────────────────────────────
# The mesh grid size. 256x256 = smooth enough, not too slow.
MESH_DOWNSAMPLE = 256
# Multiply Z (height) by this to make hills look dramatic.
# Real terrain is flat-looking in photos — exaggeration helps you see it.
HEIGHT_EXAGGERATION = 3.0

# ── SRTM (real-world elevation data) ─────────────────────────────────────────
SRTM_OPENTOPO_API = "https://opentopography.org/API/globaldem"
# Get a free API key at https://opentopography.org/ — the "demo" key has limits.
SRTM_API_KEY = ""

# ── Flask web server ──────────────────────────────────────────────────────────
FLASK_HOST = "127.0.0.1"  # only accessible from your own computer
FLASK_PORT = 5000
FLASK_DEBUG = True

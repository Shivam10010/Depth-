"""
DepthWizard — Stage 6: Flask Web Application
=============================================
The simplest possible web app:
  1. User uploads a photo (GET /)
  2. App runs depth → segment → calibrate → mesh pipeline (POST /process)
  3. App shows results: original, depth map, segmentation, height map, 3D terrain (GET / → result)

No WebSockets, no streaming, no database. Just a form → processing → results page.
"""

import os
import time
import uuid
import traceback

from flask import (
    Flask, render_template_string, request, redirect,
    url_for, flash, send_from_directory,
)

from config import OUTPUTS_DIR, FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from depth import load_model, estimate_depth
from segment import segment
from calibrate import calibrate
from mesh import build_mesh

# ── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# secret_key is needed for Flask's flash messages (error popups)
app.secret_key = "depthwizard-2025"

# Max upload size: 32 MB (aerial photos can be large TIFFs)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# Where to save uploaded images
upload_dir = os.path.join(OUTPUTS_DIR, "uploads")
os.makedirs(upload_dir, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_dir

# Allowed image formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── Lazy-load the depth model ────────────────────────────────────────────────
# Loading the model takes a few seconds and ~82 MB of RAM.
# We load it on the first request, then keep it in memory.
_depth_model = None
_depth_transform = None


def get_depth_model():
    """Load the depth model once, then reuse it for every request."""
    global _depth_model, _depth_transform
    if _depth_model is None:
        _depth_model, _depth_transform = load_model()
    return _depth_model, _depth_transform


def allowed_file(filename):
    """Check if the uploaded file is a supported image format."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


# ── HTML Templates ───────────────────────────────────────────────────────────
# We keep the HTML as Python strings so there's no separate templates/ folder.
# For a larger app you'd put these in templates/ directory.

UPLOAD_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DepthWizard — Upload</title>
    <style>
        /* ── Dark theme with purple/blue gradient ── */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: #e0e0e0;
            min-height: 100vh;
            display: flex; align-items: center; justify-content: center;
        }
        .card {
            background: #1e2a3a;
            border-radius: 16px;
            padding: 48px;
            max-width: 560px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }
        h1 {
            font-size: 2.2em; margin-bottom: 8px;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #8899aa; margin-bottom: 32px; }
        .drop-zone {
            border: 2px dashed #3a4a5a; border-radius: 12px;
            padding: 40px; text-align: center; cursor: pointer;
            transition: border-color 0.3s, background 0.3s;
        }
        .drop-zone:hover, .drop-zone.drag-over {
            border-color: #00d2ff;
            background: rgba(0, 210, 255, 0.05);
        }
        .drop-zone input { display: none; }
        .btn {
            display: inline-block; margin-top: 24px; padding: 14px 36px;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7);
            color: white; border: none; border-radius: 8px;
            font-size: 1.05em; font-weight: 600; cursor: pointer;
            text-decoration: none;
        }
        .alert {
            background: rgba(255, 80, 80, 0.15);
            border: 1px solid rgba(255, 80, 80, 0.4);
            color: #ff6b6b; padding: 12px 16px;
            border-radius: 8px; margin-bottom: 20px;
        }
        .processing { display: none; text-align: center; padding: 20px; }
        .spinner {
            width: 40px; height: 40px; margin: 0 auto 16px;
            border: 4px solid #3a4a5a; border-top-color: #00d2ff;
            border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .srtm-option {
            margin-top: 20px; padding-top: 20px;
            border-top: 1px solid #2a2a3a;
            color: #8899aa; font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>🧙 DepthWizard</h1>
        <p class="subtitle">Turn aerial/satellite images into explorable 3D terrain</p>

        <!-- Show error messages -->
        <!-- Show error messages (get_flashed_messages() is a Flask function) -->
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for msg in messages %}
                    <div class="alert">{{ msg }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <!-- Upload form -->
        <form id="form" method="post" enctype="multipart/form-data" action="/process">
            <div class="drop-zone" id="dropZone">
                <p>🛰️ Drop an aerial/satellite image here<br>or click to browse</p>
                <p id="fileName" style="color:#00d2ff; margin-top:12px;"></p>
                <input type="file" name="image" id="fileInput" accept="image/*" required>
            </div>

            <!-- SRTM checkbox for real-world heights -->
            <div class="srtm-option">
                <label>
                    <input type="checkbox" name="use_srtm" value="1"
                           style="accent-color: #00d2ff;">
                    <b>Use SRTM DEM</b> — fetch real-world elevation data (30m resolution)
                </label>
                <p style="font-size:0.8em; color:#556677; margin-top:6px;">
                    Reads GPS from image EXIF if available. Falls back to approximate heights.
                </p>
            </div>

            <div style="text-align:center;">
                <button type="submit" class="btn" id="submitBtn">Generate 3D Terrain</button>
            </div>
            <div class="processing" id="processing">
                <div class="spinner"></div>
                <p>Processing — takes 30-60 seconds on CPU...</p>
                <p style="color:#556677; font-size:0.85em; margin-top:8px;">
                    Depth estimation → Segmentation → Calibration → 3D Mesh
                </p>
            </div>
        </form>
    </div>

    <script>
        // Simple drag-and-drop + filename display
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const form = document.getElementById('form');
        const submitBtn = document.getElementById('submitBtn');
        const processing = document.getElementById('processing');

        dropZone.onclick = () => fileInput.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); };
        dropZone.ondragleave = () => dropZone.classList.remove('drag-over');
        dropZone.ondrop = (e) => {
            e.preventDefault(); dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                fileName.textContent = e.dataTransfer.files[0].name;
            }
        };
        fileInput.onchange = () => {
            if (fileInput.files.length) fileName.textContent = fileInput.files[0].name;
        };
        form.onsubmit = () => {
            submitBtn.style.display = 'none';
            processing.style.display = 'block';
        };
    </script>
</body>
</html>
"""

RESULTS_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DepthWizard — 3D Terrain</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0f1419; color: #e0e0e0;
        }
        .header {
            background: #1a1a2e; padding: 16px 32px;
            display: flex; align-items: center; justify-content: space-between;
            border-bottom: 1px solid #2a2a3a;
        }
        .header h1 {
            font-size: 1.4em;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .btn {
            padding: 8px 20px; background: #3a4a5a; color: #e0e0e0;
            border: none; border-radius: 6px; cursor: pointer;
            text-decoration: none; font-size: 0.9em;
        }
        .btn:hover { background: #4a5a6a; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px; padding: 24px; max-width: 1400px; margin: 0 auto;
        }
        .card {
            background: #1e2a3a; border-radius: 12px;
            overflow: hidden; border: 1px solid #2a2a3a;
        }
        .card-header {
            padding: 12px 16px; background: #16213e;
            font-weight: 600; font-size: 0.9em;
            color: #8899aa; text-transform: uppercase; letter-spacing: 0.5px;
        }
        .card-body {
            padding: 12px;
        }
        .card-body img { width: 100%; border-radius: 8px; display: block; }
        .card-body iframe {
            width: 100%; height: 500px; border: none; border-radius: 8px;
        }
        .info {
            padding: 8px 16px; font-size: 0.8em; color: #667788;
        }
        .full-width { grid-column: 1 / -1; }
        .warning {
            background: rgba(255, 180, 0, 0.1);
            border: 1px solid rgba(255, 180, 0, 0.3);
            color: #ffb74d; padding: 10px 16px;
            border-radius: 8px; margin: 0 24px 24px;
            max-width: 1400px; margin-left: auto; margin-right: auto;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧙 DepthWizard — 3D Terrain Model</h1>
        <a href="/" class="btn">← Upload New Image</a>
    </div>

    <!-- Show a warning if heights are approximate (no DEM used) -->
    {% if warning %}
        <div class="warning">&#9888; {{ warning }}</div>
    {% endif %}

    <div class="grid">
        <!-- Original photo -->
        <div class="card">
            <div class="card-header">&#127912; Original Image</div>
            <div class="card-body">
                <img src="/outputs/{{ upload_name }}" alt="Original">
            </div>
        </div>

        <!-- Depth heatmap -->
        <div class="card">
            <div class="card-header">&#128506; Depth Map</div>
            <div class="card-body">
                <img src="/outputs/depth_heatmap.png" alt="Depth">
                <div class="info">Brighter = closer to camera, darker = farther away</div>
            </div>
        </div>

        <!-- Segmentation mask -->
        <div class="card">
            <div class="card-header">&#127912; Segmentation</div>
            <div class="card-body">
                <img src="/outputs/segmentation_mask.png" alt="Segmentation">
                <div class="info">Gray = ground, Green = vegetation, Brown = built-up</div>
            </div>
        </div>

        <!-- Height map -->
        <div class="card">
            <div class="card-header">&#128207; Height Map</div>
            <div class="card-body">
                <img src="/outputs/height_map.png" alt="Height">
                <div class="info">{{ height_info }}</div>
            </div>
        </div>

        <!-- 3D terrain viewer -->
        <div class="card full-width">
            <div class="card-header">
                &#127956; Interactive 3D Terrain
                <span style="float:right; font-weight:400; font-size:0.8em;">
                    &#128568; Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan
                </span>
            </div>
            <div class="card-body">
                <iframe src="/outputs/terrain_3d.html" scrolling="no"></iframe>
            </div>
        </div>
    </div>
</body>
</html>
"""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Show the upload page."""
    return render_template_string(UPLOAD_PAGE)


@app.route("/process", methods=["POST"])
def process():
    """The full pipeline: upload → depth → segment → calibrate → mesh → results."""
    session_id = str(uuid.uuid4())[:8]  # short unique ID for this run
    print(f"\n{'=' * 60}")
    print(f"[app] Session {session_id}: starting processing")

    # ── Validate the upload ─────────────────────────────────────────────────
    if "image" not in request.files:
        flash("No file was uploaded.")
        return redirect(url_for("index"))

    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        flash("Please select a file first.")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported format. Use .jpg, .png, or .tif images.")
        return redirect(url_for("index"))

    # Save the uploaded file
    original_name = uploaded_file.filename
    safe_name = f"{session_id}_{original_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    uploaded_file.save(save_path)
    print(f"[app] Saved: {save_path}")

    try:
        # ── Stage 2: Estimate depth ─────────────────────────────────────────
        print("[app] Running depth estimation...")
        model, transform = get_depth_model()
        depth_normalized, _ = estimate_depth(save_path, model, transform)

        # ── Stage 3: Segment the image ──────────────────────────────────────
        print("[app] Running segmentation...")
        _, _ = segment(save_path, depth_shape=depth_normalized.shape)

        # ── Stage 4: Calibrate heights ──────────────────────────────────────
        print("[app] Calibrating heights...")
        use_srtm = request.form.get("use_srtm") == "1"
        height_map, _, source = calibrate(
            depth_normalized,
            fetch_dem=use_srtm,
            image_path=save_path,
        )

        # ── Stage 5: Build 3D mesh ──────────────────────────────────────────
        print("[app] Building 3D mesh...")
        _, _ = build_mesh(height_map, rgb_image_path=save_path)

        print(f"[app] Session {session_id}: DONE!")
        print(f"{'=' * 60}\n")

        # ── Show results page ───────────────────────────────────────────────
        # If we used a DEM, the heights are real. Otherwise, show a warning.
        used_dem = "DEM" in source
        if used_dem:
            warning = None
        else:
            warning = (
                f"Heights are approximate (0–{height_map.max():.0f}m) — "
                "no real-world elevation data was used. Results are for demo purposes."
            )

        return render_template_string(
            RESULTS_PAGE,
            upload_name=safe_name,
            warning=warning,
            height_info=f"{height_map.min():.1f}m — {height_map.max():.1f}m",
        )

    except Exception as error:
        # Print the full error for debugging
        print(f"[app] ERROR:\n{traceback.format_exc()}")
        flash(f"Processing failed: {error}")
        return redirect(url_for("index"))


# ── Static file serving ──────────────────────────────────────────────────────
# Serve the output images and HTML files from the outputs/ folder

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve files from the outputs directory."""
    return send_from_directory(OUTPUTS_DIR, filename)


# ── Run the app ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🧙 DepthWizard — open http://{FLASK_HOST}:{FLASK_PORT}")
    print("   Upload an aerial/satellite image to generate a 3D terrain model.\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

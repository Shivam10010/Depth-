"""
DepthWizard — Stage 5: 3D Mesh & Interactive Viewer
====================================================
Takes the height map and creates an interactive 3D terrain you can
rotate, zoom, and explore in a web browser.

────────────────── WHAT'S HAPPENING ──────────────────
1. Downsample the height map to 256×256 (fewer points = smoother 3D rendering)
2. Build X, Y grid coordinates for each point
3. Multiply Z (height) by HEIGHT_EXAGGERATION so terrain looks dramatic
4. Generate an HTML file with Plotly (JavaScript charting library)
   that renders the 3D surface in any browser — no server needed!
"""

import os
import json
import numpy as np
import cv2
import plotly.graph_objects as go
from config import OUTPUTS_DIR, MESH_DOWNSAMPLE, HEIGHT_EXAGGERATION


def build_mesh(height_map, rgb_image_path=None,
               downsample=MESH_DOWNSAMPLE,
               height_exaggeration=HEIGHT_EXAGGERATION):
    """Create a 3D terrain mesh and save an interactive HTML viewer.

    Args:
        height_map: 2D numpy array — heights in meters (from Stage 4)
        rgb_image_path: optional path to the original photo (for texture)
        downsample: grid resolution (256 = good balance of smooth + fast)
        height_exaggeration: multiply Z by this (3.0 = 3x taller hills)

    Returns:
        mesh_data: dict with X, Y, Z arrays and metadata
        html_path: where the interactive HTML was saved
    """
    print(f"[mesh] Building {downsample}x{downsample} mesh "
          f"with {height_exaggeration}x height exaggeration...")

    # ── Step 1: Downsample to a manageable grid size ───────────────────────
    # A 1024×1024 height map has 1 million points — too many for smooth 3D.
    # 256×256 = 65,536 points — smooth enough and renders instantly.
    height_down = _shrink_to_square(height_map, downsample)
    rows, cols = height_down.shape

    # ── Step 2: Create X, Y coordinate grids ───────────────────────────────
    # Think of it like graph paper: x goes left→right, y goes bottom→top
    x_coords = np.arange(cols, dtype=np.float32)
    y_coords = np.arange(rows, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Flip Y so the image's top row = the terrain's "back" (standard for 3D)
    y_grid = np.flipud(y_grid)

    # ── Step 3: Apply height exaggeration ──────────────────────────────────
    # Real terrain from aerial photos looks very flat — multiplying by 3x
    # makes the hills actually visible in 3D.
    z_exaggerated = height_down * height_exaggeration

    # ── Step 4: Collect mesh data ──────────────────────────────────────────
    mesh_data = {
        "x": x_grid.tolist(),
        "y": y_grid.tolist(),
        "z": height_down.tolist(),
        "z_exaggerated": z_exaggerated.tolist(),
        "rows": rows,
        "cols": cols,
        "height_exaggeration": float(height_exaggeration),
        "height_range": [float(height_down.min()), float(height_down.max())],
    }

    # ── Step 5: Load RGB texture (optional) ────────────────────────────────
    rgb_texture = None
    if rgb_image_path and os.path.exists(rgb_image_path):
        try:
            image_bgr = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            rgb_texture = _shrink_to_square(image_rgb, downsample)
            rgb_texture = np.flipud(rgb_texture)  # flip to match the mesh
            mesh_data["texture_rgb"] = rgb_texture.tolist()
            print("[mesh] RGB texture loaded.")
        except Exception as error:
            print(f"[mesh] Couldn't load texture: {error}")

    # ── Step 6: Generate the interactive HTML ──────────────────────────────
    html_path = os.path.join(OUTPUTS_DIR, "terrain_3d.html")
    _write_plotly_html(z_exaggerated, height_down, rows, cols, mesh_data, html_path)

    # ── Step 7: Save raw mesh data as JSON (for Three.js upgrades, etc.) ──
    json_path = os.path.join(OUTPUTS_DIR, "mesh_data.json")
    with open(json_path, "w") as f:
        json.dump(mesh_data, f)

    print(f"[mesh] Saved interactive viewer: {html_path}")
    print(f"[mesh] Saved mesh data: {json_path}")

    return mesh_data, html_path


def _shrink_to_square(image_array, target_size):
    """Resize a 2D or 3D (H×W or H×W×C) array to target_size × target_size."""
    height, width = image_array.shape[:2]
    if height == target_size and width == target_size:
        return image_array

    return cv2.resize(
        image_array.astype(np.float32),
        (target_size, target_size),
        interpolation=cv2.INTER_AREA,
    )


def _write_plotly_html(z_exag, z_orig, rows, cols, mesh_data, output_path):
    """Generate an HTML file with an interactive Plotly 3D surface plot.

    Plotly is a JavaScript library for interactive charts.
    We use `go.Surface` to draw a 3D terrain.

    The HTML is completely self-contained — open it in any browser
    and you can drag to rotate, scroll to zoom.
    """
    # Terrain colors: deep blue → green → yellow → brown → white
    terrain_colors = [
        [0.0,  "#1a3a5c"],
        [0.15, "#2d6a4f"],
        [0.35, "#52b788"],
        [0.50, "#d4e09b"],
        [0.65, "#f4d35e"],
        [0.80, "#c97a4b"],
        [0.92, "#8b5e3c"],
        [1.0,  "#f5f0e8"],
    ]

    figure = go.Figure()

    # The 3D surface — each point's height comes from z_exag
    figure.add_trace(go.Surface(
        z=z_exag,
        colorscale=terrain_colors,
        colorbar=dict(
            title="Elevation (m)",
            tickfont=dict(size=10, color="#aaa"),
            len=0.65,
            thickness=18,
            x=1.02,
        ),
        # Lighting makes the terrain look 3D (shading from a "sun" direction)
        lighting=dict(
            ambient=0.4,    # overall brightness
            diffuse=0.7,    # light that bounces off the surface
            roughness=0.5,  # how matte/shiny the surface is
            specular=0.2,   # shiny highlights
            fresnel=0.15,   # edge glow effect
        ),
        lightposition=dict(x=3, y=2, z=4),
        # Show contour lines on the surface
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="white",
                highlightwidth=1.5,
                project={"z": True},
            )
        ),
        name="Terrain",
        # When you hover over a point, show its elevation
        hovertemplate=(
            "Position: (%{x:.0f}, %{y:.0f})<br>"
            "Elevation: %{customdata:.1f}m<br>"
            "<extra></extra>"
        ),
        customdata=z_orig,
    ))

    # Layout: dark background, labeled axes, nice camera angle
    h_min, h_max = mesh_data.get("height_range", [0, 50])
    figure.update_layout(
        title=dict(
            text="<b>DepthWizard — 3D Terrain Model</b>",
            x=0.5,  # center the title
            xanchor="center",
            font=dict(size=20, color="#e0e0e0"),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        scene=dict(
            xaxis=dict(title="X (px)", showgrid=True),
            yaxis=dict(title="Y (px)", showgrid=True),
            zaxis=dict(
                title=f"Elevation (m) — {h_min:.0f}m to {h_max:.0f}m",
                showgrid=True,
                range=[0, h_max * HEIGHT_EXAGGERATION * 1.05],
            ),
            # Start with a nice angled view (not straight on)
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
            bgcolor="#0d1117",
            aspectmode="cube",  # keep X, Y, Z axes the same scale
        ),
        margin=dict(l=0, r=30, t=50, b=0),
        autosize=True,
    )

    # Write self-contained HTML (Plotly JS loaded from CDN)
    figure.write_html(
        output_path,
        include_plotlyjs="cdn",  # don't bundle Plotly — load from internet
        auto_open=False,         # don't open a browser tab (we're in Flask)
        config=dict(
            responsive=True,
            displayModeBar=True,   # show zoom/pan/screenshot buttons
            displaylogo=False,
            scrollZoom=True,
        ),
    )

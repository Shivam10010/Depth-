"""
DepthWizard — Stage 4: Height Calibration
==========================================
Converts the relative depth map (0 to 1) into real-world heights (meters).

────────────────── THE PROBLEM ──────────────────
The depth model only knows "this pixel is closer than that pixel" —
it doesn't know "this building is 30 meters tall."

────────────────── THE SOLUTIONS ─────────────────
We have three options, in order of preference:

1. USE A DEM (Digital Elevation Model):
   A DEM is a map of real-world ground elevations from satellite data.
   If you have one (e.g., SRTM 30m data), we can line up the predicted
   depth with the real elevations and get metric heights.

2. AUTO-FETCH SRTM:
   If the user checks "Use SRTM DEM" in the web app, we try to read
   GPS coordinates from the photo's EXIF data and download the
   matching SRTM tile automatically.

3. SIMPLE SCALING (fallback):
   If no DEM is available, we just say "the deepest pixel = max_height meters"
   and linearly scale everything. This gives approximate, demo-quality results.
"""

import os
import numpy as np
import cv2
from config import OUTPUTS_DIR, DEFAULT_MAX_HEIGHT_M


def calibrate(depth_norm, dem_path=None, max_height=DEFAULT_MAX_HEIGHT_M,
              fetch_dem=False, image_path=None):
    """Turn relative depth (0-1) into heights in meters.

    Args:
        depth_norm: 2D array, values from 0.0 to 1.0 (output of Stage 2)
        dem_path: optional path to a DEM GeoTIFF file for real calibration
        max_height: max height in meters (used for simple scaling fallback)
        fetch_dem: if True, try to auto-download SRTM data
        image_path: original image path (used to read GPS EXIF for SRTM)

    Returns:
        height_map: 2D array of heights in meters
        viz_path: path to the saved height visualization PNG
        source: string describing which calibration method was used
    """
    dem_file = _resolve_dem(dem_path, fetch_dem, image_path)

    if dem_file is not None:
        height_map, source = _calibrate_with_dem(depth_norm, dem_file, max_height)
    else:
        height_map, source = _calibrate_simple(depth_norm, max_height)

    # ── Make a pretty visualization ─────────────────────────────────────────
    viz_path = os.path.join(OUTPUTS_DIR, "height_map.png")
    _save_height_visual(height_map, viz_path)

    h_min = float(height_map.min())
    h_max = float(height_map.max())
    print(f"[calibrate] {source}")
    print(f"[calibrate] Height range: {h_min:.1f}m — {h_max:.1f}m")

    return height_map, viz_path, source


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_dem(dem_path, fetch_dem, image_path):
    """Figure out which DEM file to use, if any.

    Returns the path to a DEM file, or None if we should use simple scaling.
    """
    # Option 1: user provided a DEM file
    if dem_path and os.path.exists(dem_path):
        print(f"[calibrate] Using provided DEM: {dem_path}")
        return dem_path

    # Option 2: try to auto-fetch SRTM from the image's GPS data
    if fetch_dem and image_path:
        print("[calibrate] Trying to auto-fetch SRTM DEM...")
        dem_file = _try_fetch_srtm(image_path)
        if dem_file:
            return dem_file
        print("[calibrate] SRTM fetch failed — will use simple scaling.")

    return None


def _calibrate_with_dem(depth_norm, dem_path, max_height):
    """Use a real DEM to calibrate heights.

    Strategy: resize the DEM to match the depth map size, then find the
    min/max elevation in the DEM. Scale the depth values so they span
    the same range as the DEM.

    This works because:
    - Where the DEM says "500m elevation", the depth model should say
      "this pixel is higher than average"
    - Where the DEM says "550m elevation", the depth model should say
      "even higher"
    """
    try:
        import rasterio
        from rasterio.warp import resize as rio_resize

        print(f"[calibrate] Reading DEM: {dem_path}")

        # Read the DEM (it's a single-band GeoTIFF — just elevation numbers)
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)  # band 1 = elevation

        # Resize DEM to match our depth map dimensions
        depth_height, depth_width = depth_norm.shape[:2]
        dem_resized = rio_resize(
            dem_data,
            out_shape=(depth_height, depth_width),
            resampling=rasterio.enums.Resampling.bilinear,
        )

        # Find valid (non-NaN) elevation values
        valid_mask = np.isfinite(dem_resized)
        if not valid_mask.any():
            raise ValueError("DEM has no valid data")

        dem_min = dem_resized[valid_mask].min()
        dem_max = dem_resized[valid_mask].max()
        dem_range = dem_max - dem_min

        # Scale depth values to match DEM elevation range
        if dem_range > 1e-8:
            height_map = dem_min + depth_norm * dem_range
        else:
            # Flat terrain (all same elevation)
            height_map = np.full_like(depth_norm, dem_min)

        source = f"DEM-calibrated ({dem_min:.1f}m – {dem_max:.1f}m)"
        print(f"[calibrate] DEM elevation range: {dem_min:.1f}m – {dem_max:.1f}m")
        return height_map, source

    except ImportError:
        print("[calibrate] rasterio not installed — falling back to simple scaling.")
        return _calibrate_simple(depth_norm, max_height)
    except Exception as error:
        print(f"[calibrate] DEM failed ({error}) — falling back to simple scaling.")
        return _calibrate_simple(depth_norm, max_height)


def _calibrate_simple(depth_norm, max_height):
    """Just multiply depth by max_height.
    e.g., if max_height=50, depth 0.0 → 0m, depth 0.5 → 25m, depth 1.0 → 50m.
    This is approximate — real heights could be anything.
    """
    height_map = depth_norm * max_height
    source = f"approximate (0–{max_height}m, no real elevation data)"
    return height_map, source


def _try_fetch_srtm(image_path):
    """Try to download SRTM DEM tiles based on GPS data in the image.

    SRTM = Shuttle Radar Topography Mission — ~30m resolution elevation data.
    """
    try:
        from srtm_downloader import estimate_latlon_from_image, fetch_dem_for_bounds

        # Try to read GPS coordinates from the image
        bounds = estimate_latlon_from_image(image_path, None)
        if bounds is None:
            print("[calibrate] No GPS data found in image.")
            return None

        min_lat, max_lat, min_lon, max_lon = bounds
        print(f"[calibrate] Image location: [{min_lat:.4f}, {min_lon:.4f}] → "
              f"[{max_lat:.4f}, {max_lon:.4f}]")

        # Download the SRTM tile(s) for this area
        return fetch_dem_for_bounds(min_lat, max_lat, min_lon, max_lon)

    except ImportError:
        print("[calibrate] srtm_downloader module not found.")
    except Exception as error:
        print(f"[calibrate] SRTM download failed: {error}")
    return None


# ── Visualization helpers ─────────────────────────────────────────────────────

def _save_height_visual(height_map, output_path):
    """Save a color-coded height map image.

    Uses a terrain-like colormap: blue (low) → green → yellow → brown → white (high).
    """
    # Normalize to 0-255 for display
    h_min = height_map.min()
    h_max = height_map.max()
    h_range = h_max - h_min

    if h_range > 1e-8:
        normalized = ((height_map - h_min) / h_range * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(height_map, dtype=np.uint8)

    # Apply a terrain colormap (lookup table approach — fast and reliable)
    colormap = _make_terrain_colors()
    colored_image = _apply_colormap(normalized, colormap)

    cv2.imwrite(output_path, colored_image)


def _make_terrain_colors():
    """Create a 256-color terrain colormap.

    Returns a numpy array of shape (256, 3) in BGR order (for OpenCV).
    Color progression: deep blue → green → yellow → brown → white
    """
    # Define color stops: (position, RGB color)
    # Matplotlib's "terrain" colormap style
    color_stops = [
        (0.00, (26,  58,  90)),    # deep blue   — lowest
        (0.15, (45, 106,  79)),    # dark green
        (0.35, (82, 183, 120)),    # green
        (0.50, (212, 224, 155)),   # light green/yellow
        (0.65, (244, 211, 94)),    # yellow
        (0.80, (201, 122,  75)),   # brown
        (0.90, (139,  94,  60)),   # dark brown
        (1.00, (245, 240, 232)),   # white       — highest
    ]

    colors_rgb = []
    for position, rgb in color_stops:
        colors_rgb.append((position, rgb))

    # Interpolate between stops to get all 256 colors
    positions = [c[0] for c in colors_rgb]
    r_values = [c[1][0] for c in colors_rgb]
    g_values = [c[1][1] for c in colors_rgb]
    b_values = [c[1][2] for c in colors_rgb]

    # Create 256 evenly-spaced positions
    query_positions = np.linspace(0, 1, 256)

    # Interpolate each channel
    r_interp = np.interp(query_positions, positions, r_values)
    g_interp = np.interp(query_positions, positions, g_values)
    b_interp = np.interp(query_positions, positions, b_values)

    # Stack into (256, 3) array and convert RGB → BGR for OpenCV
    colormap_rgb = np.stack([r_interp, g_interp, b_interp], axis=1).astype(np.uint8)
    colormap_bgr = colormap_rgb[:, ::-1]  # flip R and B channels
    return colormap_bgr


def _apply_colormap(gray_image, colormap):
    """Replace each gray pixel with the corresponding color from the colormap.

    This is like OpenCV's applyColorMap but using our custom terrain colors.
    """
    # gray_image values (0-255) are indices into the colormap
    return colormap[gray_image]

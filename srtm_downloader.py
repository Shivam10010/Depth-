"""
DepthWizard — SRTM 30m DEM Downloader.
Fetches SRTM 30m tiles from OpenTopography and mosaics them into a single DEM.
Requires: rasterio (optional, for reading the downloaded GeoTIFF).
"""
import os
import math
import requests
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling
import numpy as np
from config import OUTPUTS_DIR


def _latlon_to_srtm_tile(lat, lon):
    """Convert lat/lon to SRTM 1-arc-second (30m) tile naming convention.

    SRTM naming: N{lat:02d}E{lon:03d} or N{lat:02d}W{lon:03d}
    """
    lat_idx = int(math.floor(lat))
    lon_idx = int(math.floor(lon))
    ns = "N" if lat_idx >= 0 else "S"
    ew = "E" if lon_idx >= 0 else "W"
    return f"{ns}{abs(lat_idx):02d}{ew}{abs(lon_idx):03d}"


def download_srtm_tile(tile_name, output_dir=None):
    """Download a single SRTM 30m tile from OpenTopography.

    Args:
        tile_name: SRTM tile name like "N17E073" or "N30W090".
        output_dir: Directory to save the GeoTIFF (default: outputs/).

    Returns:
        Path to the downloaded GeoTIFF, or None on failure.
    """
    output_dir = output_dir or OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # OpenTopography SRTM 30m API (requires free API key from opentopography.org)
    # The tile name format: N17E073
    # Try multiple URLs for robustness
    urls = [
        f"https://opentopography.org/API/globaldem?demtype=SRTMGL1&south={tile_name[1:3] if tile_name[0]=='N' else '-'+tile_name[1:3]}&north={int(tile_name[1:3])+1 if tile_name[0]=='N' else int(tile_name[1:3])-1}&west={int(tile_name[4:7])}&east={int(tile_name[4:7])+1}&outputFormat=GTiff&API_Key=demo",
    ]

    output_path = os.path.join(output_dir, f"srtm_{tile_name}.tif")

    if os.path.exists(output_path):
        print(f"[srtm] Tile already cached: {output_path}")
        return output_path

    for url in urls:
        try:
            print(f"[srtm] Downloading tile {tile_name} from {url[:60]}...")
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and len(resp.content) > 10000:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                print(f"[srtm] Saved → {output_path} ({len(resp.content) // 1024} KB)")
                return output_path
            else:
                print(f"[srtm] HTTP {resp.status_code} for {tile_name}")
        except Exception as e:
            print(f"[srtm] Failed to download {tile_name}: {e}")

    print(f"[srtm] Could not download tile {tile_name}")
    return None


def fetch_dem_for_bounds(min_lat, max_lat, min_lon, max_lon, output_dir=None):
    """Fetch SRTM DEM tiles covering a bounding box and merge them.

    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box in decimal degrees.
        output_dir: Where to save the merged DEM.

    Returns:
        Path to merged DEM GeoTIFF, or None on failure.
    """
    output_dir = output_dir or OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Determine which tiles we need
    lat_tiles = range(int(math.floor(min_lat)), int(math.floor(max_lat)) + 1)
    lon_tiles = range(int(math.floor(min_lon)), int(math.floor(max_lon)) + 1)

    tile_names = []
    for lat in lat_tiles:
        for lon in lon_tiles:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile_names.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")

    print(f"[srtm] Fetching {len(tile_names)} SRTM tile(s) for bbox "
          f"[{min_lat:.2f},{min_lon:.2f}] → [{max_lat:.2f},{max_lon:.2f}] ...")

    # Download tiles
    tile_files = []
    for tile_name in tile_names:
        path = download_srtm_tile(tile_name, output_dir)
        if path:
            tile_files.append(path)

    if not tile_files:
        print("[srtm] No tiles downloaded. Falling back to approximate calibration.")
        return None

    if len(tile_files) == 1:
        return tile_files[0]

    # Merge multiple tiles
    print(f"[srtm] Merging {len(tile_files)} tiles...")
    try:
        srcs = [rasterio.open(f) for f in tile_files]
        mosaic, out_transform = merge(srcs, resampling=Resampling.bilinear)

        # Use metadata from the first source
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
        })

        merged_path = os.path.join(output_dir, "srtm_merged.tif")
        with rasterio.open(merged_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        for src in srcs:
            src.close()

        print(f"[srtm] Merged DEM saved → {merged_path}")
        return merged_path

    except Exception as e:
        print(f"[srtm] Merge failed ({e}) — using first tile.")
        return tile_files[0]


def estimate_latlon_from_image(image_path, depth_shape):
    """Estimate approximate lat/lon bounds from an image filename or EXIF.

    This is a placeholder — for demo purposes, you can hardcode coordinates.
    For production, parse EXIF GPS data with PIL.ExifTags.

    Args:
        image_path: Path to the uploaded image.
        depth_shape: Shape of the depth map (h, w).

    Returns:
        (min_lat, max_lat, min_lon, max_lon) or None if unknown.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None

        # Try to find GPS info
        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                gps_info = {GPSTAGS.get(t, t): v for t, v in value.items()}

        if not gps_info:
            return None

        # Extract lat/lon from GPS
        def _dms_to_dd(dms, ref):
            d, m, s = dms
            dd = d + m / 60 + s / 3600
            if ref in ("S", "W"):
                dd = -dd
            return dd

        lat = _dms_to_dd(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
        lon = _dms_to_dd(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))

        # Add a small buffer based on image resolution
        # Approximate: 1 pixel ≈ ground sample distance
        # For demo, use ~0.001 degree ≈ 111m
        buffer = 0.005
        return (lat - buffer, lat + buffer, lon - buffer, lon + buffer)

    except Exception:
        return None

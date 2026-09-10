# DepthWizard

Turn aerial/satellite RGB images into explorable 3D terrain models.

> **Hackathon prototype** — reliability and visual impact first. Runs on CPU only.

---

## Quick Start (3 commands)

```bash
cd "c:\Users\itsvi\OneDrive\Desktop\depth 2nd"
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Then open **http://127.0.0.1:5000** in a browser, upload an image, and explore the 3D terrain.

---

## Pipeline Stages

| Stage | Module | What it does |
|-------|--------|-------------|
| 2 | [depth.py](depth.py) | MiDaS_small → relative depth map |
| 3 | [segment.py](segment.py) | OpenCV color heuristics → vegetation / built-up / ground masks |
| 4 | [calibrate.py](calibrate.py) | Scale depth → real-world meters (or align with SRTM/DEM GeoTIFF) |
| 5 | [mesh.py](mesh.py) | 256×256 grid mesh with exaggeration + contour lines → Plotly HTML |
| 6 | [app.py](app.py) | Flask web app: upload → process → 3D viewer |

### Switching depth models

Change one line in [config.py](config.py):

```python
DEPTH_MODEL_NAME = "MiDaS_small"    # CPU-friendly, reliable
# DEPTH_MODEL_NAME = "dpt_hybrid"   # better quality, slower
```

---

## Outputs

Each run saves intermediates to `outputs/`:

- `depth_heatmap.png` — MiDaS inferred depth (Inferno colormap)
- `segmentation_mask.png` — vegetation (green) / built-up (brown) / ground (gray)
- `height_map.png` — calibrated elevation in meters (terrain colormap)
- `terrain_3d.html` — interactive 3D surface (drag to rotate, scroll to zoom)
- `mesh_data.json` — raw mesh data for optional Three.js upgrade

---

## Real-World Heights (SRTM DEM)

Check the **"Use SRTM DEM"** checkbox on the upload page to fetch SRTM 30m tiles automatically:

1. Reads GPS coordinates from image EXIF (if present)
2. Downloads SRTM 30m tiles via OpenTopography API
3. Scales depth to match the real DEM elevation range for that area

For manual DEM calibration, provide a GeoTIFF path. See [srtm_downloader.py](srtm_downloader.py) for the API.

---

## Notes

- **Heights are approximate** without a DEM — scaled 0–50m by default.
- **Mesh:** 256×256 grid with 3× height exaggeration for dramatic terrain visibility.
- Runs entirely on CPU. MiDaS_small processes a 512×512 image in ~0.2s.
- First run downloads model weights (~82 MB) via `torch.hub` and caches them.

---

## Demo Reliability Guide

| Component | Status |
|-----------|--------|
| MiDaS_small depth estimation | Core, tested, reliable |
| OpenCV segmentation | Core, tested, reliable |
| Height calibration (scale mode) | Core, tested, reliable |
| SRTM DEM auto-fetch (EXIF GPS) | Optional — requires GPS-tagged image |
| DEM calibration (manual GeoTIFF) | Optional — requires `rasterio` |
| Plotly 3D viewer (terrain colormap, contours) | Core, tested, reliable |
| Flask upload app | Core, tested, reliable |
| Three.js upgrade path | Optional stretch — mesh_data.json is ready |

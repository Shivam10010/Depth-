"""
DepthWizard — Stage-by-stage verification script.
Run individual tests:  python test_stages.py 2  (Stage 2 only)
                       python test_stages.py 2 3  (Stages 2 + 3)
                       python test_stages.py all  (all stages)

Requires: a test image at inputs/sample_image.jpg (or pass --image path).
"""
import sys
import os
import argparse
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Default test image location
DEFAULT_TEST_IMAGE = os.path.join(BASE_DIR, "inputs", "sample_image.jpg")


def generate_test_image(path=DEFAULT_TEST_IMAGE, size=(512, 512)):
    """Create a synthetic aerial-like image if no test image exists."""
    import numpy as np
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)

    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient (top half)
    for y in range(h // 2):
        t = y / (h // 2)
        img[y, :] = [int(135 * (1 - t) + 180 * t),
                     int(180 * (1 - t) + 210 * t),
                     int(220 * (1 - t) + 240 * t)]

    # Ground: green-brown gradient (bottom half)
    for y in range(h // 2, h):
        t = (y - h // 2) / (h // 2)
        img[y, :] = [int(60 + 40 * t),
                     int(100 + 60 * t),
                     int(40 + 20 * t)]

    # Add some "building" rectangles (gray-blue)
    np.random.seed(42)
    for _ in range(6):
        bx = np.random.randint(50, w - 150)
        by = np.random.randint(h // 2 + 30, h - 60)
        bw = np.random.randint(40, 100)
        bh = np.random.randint(30, 70)
        shade = np.random.randint(140, 200)
        img[by:by + bh, bx:bx + bw] = [shade - 20, shade - 10, shade + 10]

    # Add "roads" (lighter strips)
    road_y = h // 2 + 120
    img[road_y:road_y + 8, :] = [160, 155, 150]

    # Add some "vegetation" patches (green blobs)
    for _ in range(8):
        cx = np.random.randint(20, w - 20)
        cy = np.random.randint(h // 2 + 20, h - 20)
        radius = np.random.randint(15, 35)
        cv2.circle(img, (cx, cy), radius, (34, 100, 30), -1)

    cv2.imwrite(path, img)
    print(f"[test] Generated synthetic test image → {path}  ({w}x{h})")
    return path


def test_stage2(image_path):
    """Test depth estimation."""
    print("\n" + "=" * 60)
    print("TEST — Stage 2: Depth Estimation")
    print("=" * 60)

    from depth import estimate_depth, load_model

    print("  Loading model (first run downloads ~40 MB)...")
    t0 = time.time()
    model, transform = load_model()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    print(f"  Running depth estimation on {image_path} ...")
    t0 = time.time()
    depth_norm, viz_path = estimate_depth(image_path, model, transform)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    assert depth_norm is not None, "Depth array is None"
    assert depth_norm.dtype == np.float32, f"Wrong dtype: {depth_norm.dtype}"
    assert depth_norm.min() >= 0.0 and depth_norm.max() <= 1.0, "Depth not in [0,1]"
    assert os.path.exists(viz_path), f"Heatmap not saved: {viz_path}"

    print(f"  ✅ PASS — depth shape: {depth_norm.shape}, range: [{depth_norm.min():.3f}, {depth_norm.max():.3f}]")
    print(f"  ✅ Heatmap saved: {viz_path}")
    return depth_norm


def test_stage3(image_path, depth_shape=None):
    """Test segmentation."""
    print("\n" + "=" * 60)
    print("TEST — Stage 3: Segmentation")
    print("=" * 60)

    from segment import segment

    labels, mask_path = segment(image_path, depth_shape=depth_shape)

    assert labels is not None, "Labels array is None"
    assert labels.ndim == 2, f"Expected 2D labels, got {labels.ndim}D"
    unique_labels = set(np.unique(labels))
    assert unique_labels.issubset({0, 1, 2}), f"Unexpected labels: {unique_labels}"
    assert os.path.exists(mask_path), f"Mask not saved: {mask_path}"

    counts = {label: int((labels == label).sum()) for label in [0, 1, 2]}
    names = {0: "ground", 1: "vegetation", 2: "built-up"}
    for label, count in counts.items():
        pct = count / labels.size * 100
        print(f"  {names[label]}: {count} px ({pct:.1f}%)")

    print(f"  ✅ PASS — mask shape: {labels.shape}, labels: {unique_labels}")
    print(f"  ✅ Mask saved: {mask_path}")
    return labels


def test_stage4(depth_norm):
    """Test height calibration."""
    print("\n" + "=" * 60)
    print("TEST — Stage 4: Height Calibration (no DEM)")
    print("=" * 60)

    from calibrate import calibrate

    height_map, viz_path, source = calibrate(depth_norm)

    assert height_map is not None, "Height map is None"
    assert height_map.shape == depth_norm.shape, "Shape mismatch"
    assert height_map.min() >= 0, "Negative heights"
    assert os.path.exists(viz_path), f"Height viz not saved: {viz_path}"

    print(f"  Source: {source}")
    print(f"  ✅ PASS — height range: {height_map.min():.1f}m – {height_map.max():.1f}m")
    print(f"  ✅ Height map saved: {viz_path}")
    return height_map


def test_stage5(height_map, image_path):
    """Test mesh generation + Plotly HTML."""
    print("\n" + "=" * 60)
    print("TEST — Stage 5: Mesh + Visualization")
    print("=" * 60)

    from mesh import build_mesh

    mesh_data, viz_path = build_mesh(height_map, rgb_image_path=image_path)

    assert "x" in mesh_data, "Missing x in mesh data"
    assert "y" in mesh_data, "Missing y in mesh data"
    assert "z" in mesh_data, "Missing z in mesh data"
    assert os.path.exists(viz_path), f"HTML not saved: {viz_path}"

    # Verify HTML file has content
    size = os.path.getsize(viz_path)
    assert size > 1000, f"HTML file suspiciously small: {size} bytes"

    # Verify grid dimensions
    z = mesh_data["z"]
    assert len(z) == mesh_data["rows"], "Row count mismatch"
    assert len(z[0]) == mesh_data["cols"], "Col count mismatch"

    print(f"  Mesh grid: {mesh_data['rows']}×{mesh_data['cols']}")
    print(f"  ✅ PASS — Plotly HTML saved: {viz_path} ({size:,} bytes)")
    print(f"  Open in browser: file:///{viz_path.replace(os.sep, '/')}")
    return mesh_data


def test_stage6():
    """Test Flask app via test client."""
    print("\n" + "=" * 60)
    print("TEST — Stage 6: Flask App")
    print("=" * 60)

    from app import app as flask_app

    client = flask_app.test_client()

    # Test GET /
    resp = client.get("/")
    assert resp.status_code == 200, f"GET / returned {resp.status_code}"
    assert b"DepthWizard" in resp.data, "Index page missing title"
    print("  GET / → 200 OK")

    # Test 404 for outputs
    resp = client.get("/outputs/nonexistent.png")
    assert resp.status_code == 404
    print("  GET /outputs/nonexistent → 404 (correct)")

    print("  ✅ PASS — Flask app routes working")


def main():
    parser = argparse.ArgumentParser(description="DepthWizard stage-by-stage tests")
    parser.add_argument("stages", nargs="+", help="Stage numbers (2-6) or 'all'")
    parser.add_argument("--image", default=None, help="Path to test image")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip generating a test image if none exists")
    args = parser.parse_args()

    # Resolve test image
    image_path = args.image or DEFAULT_TEST_IMAGE
    if not os.path.exists(image_path) and not args.skip_generate:
        image_path = generate_test_image()

    if not os.path.exists(image_path):
        print(f"ERROR: No test image found at {image_path}")
        print("  Provide one with --image or let the script generate one.")
        sys.exit(1)

    print(f"Using test image: {image_path}")

    stages = args.stages
    if "all" in stages:
        stages = [2, 3, 4, 5, 6]

    stages = [int(s) for s in stages]

    depth_norm = None
    labels = None
    height_map = None

    for s in stages:
        if s == 2:
            depth_norm = test_stage2(image_path)
        elif s == 3:
            depth_shape = depth_norm.shape if depth_norm is not None else None
            labels = test_stage3(image_path, depth_shape)
        elif s == 4:
            if depth_norm is None:
                print("WARNING: Running Stage 4 without Stage 2 — generating depth first")
                depth_norm = test_stage2(image_path)
            height_map = test_stage4(depth_norm)
        elif s == 5:
            if height_map is None:
                if depth_norm is None:
                    depth_norm = test_stage2(image_path)
                labels = test_stage3(image_path, depth_norm.shape) if labels is None else labels
                height_map = test_stage4(depth_norm)
            test_stage5(height_map, image_path)
        elif s == 6:
            test_stage6()
        else:
            print(f"Unknown stage: {s} (valid: 2-6, all)")

    print("\n" + "=" * 60)
    print("ALL REQUESTED STAGES PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

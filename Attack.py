from PIL import Image
from collections import deque, defaultdict
import random
import math
from scipy.optimize import brentq
import contextlib
import json
import time
from geopy.distance import geodesic
import numpy as np
import os
import sys
import pyproj
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import xyzservices.providers as xyz
from difflib import get_close_matches

"""
Generate one configured map, run the modified perceptual-descent method,
and write evaluation artifacts for that run.

Usage:
1. Set the run parameters in the config sections below, especially
   `RUN_DATASET`, `TEST_NAME`, and the rendering / optimization settings.
2. Run the script with `python Attack.py`.
3. Check the generated outputs in `Results/<TEST_NAME>/<dataset type>/`.

 Dataset options:
- `OpenAddresses` uses `CoordinateJSONs/OpenAddress/US.geojson`
- `Synthetic` uses `CoordinateJSONs/Synthetic/US.geojson`
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "CoordinateJSONs")

# =========================
# Run Selection
# =========================
RUN_DATASET = {
    "OpenAddresses": "OpenAddress/US.geojson",
    "Synthetic": "Synthetic/US.geojson",
}  # Datasets to run, mapping dataset type to its GeoJSON file.
TEST_NAME = "192dpiSatelliteBG-US"  # Match the original script default.
EVAL_DECIMALS = 6  # Decimal places used when rounding evaluation coordinates.
CLUSTER_TYPE = "new"
CLUSTER_SIZE_MODE = "estimate"  # "manual" or "estimate"

# =========================
# Rendering
# =========================
BG_MODE = False  # Use background-reference images when comparing pixel changes.
REGENERATE_BASE_MAP = True  # Rebuild the main input map even if the PNG already exists.
WIDTH_PX = 2284  # Output map width in pixels.
HEIGHT_PX = 1424  # Output map height in pixels.
DOT_RADIUS_MM = 2  # Dot radius in millimeters when the map is rendered.
PRIMARY_TILE_SOURCE = xyz.OpenStreetMap.Mapnik  # Basemap used for the original rendered map.
SHIFTED_TILE_SOURCE = xyz.Esri.WorldStreetMap  # Basemap used for candidate comparison renders.

# =========================
# Optimization
# =========================
MAX_ITER = 100  # Max k-means refinement iterations per blob.
TOL = 1e-2  # Center-movement tolerance used to stop blob refinement.
INITIAL_STEP_SIZE = 0.5  # Starting pixel step size for perceptual descent.
STEP_DIVISOR = 1.2  # Amount to shrink the step size after each descent round.
MIN_STEP_SIZE = 0.0001  # Stop descent once the step size falls below this threshold.

# =========================
# Map Bounds
# =========================
PIXEL_SIZE = 0.02587884152408056  # Degrees represented by one pixel in the rendered map.
MIN_LON = -126.17658145147592563  # Western map boundary in longitude.
MAX_LAT = 58.62037301762128294  # Northern map boundary in latitude.
MAP_ZOOM = 5  # Basemap zoom level used during rendering.
# Coordinates are defaults for US Map. The configurations for other datasets 
# should be set to tightly frame the true points with some margin.

EVAL_SOURCE_FILES = {
    "OpenAddresses": "OpenAddress/US.geojson",
    "Synthetic": "Synthetic/US.geojson",
}  # GeoJSON used as ground truth when scoring each dataset.

RUN_SUFFIX = f"_{TEST_NAME}" if TEST_NAME else ""
RESULTS_ROOT = os.path.join(BASE_DIR, "Results")
AUGMENTED_ROOT = os.path.join(BASE_DIR, "AugmentedFiles")
CLUSTER_TYPE_ROOT = os.path.join(AUGMENTED_ROOT, "cluster_types")
os.makedirs(CLUSTER_TYPE_ROOT, exist_ok=True)

DOT_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)
CURRENT_DATASET_KEY = next(iter(RUN_DATASET))
JSON_FILE = RUN_DATASET[CURRENT_DATASET_KEY]
EVAL_SOURCE_FILE = ""
RESULTS_RUN_DIR = ""
AUGMENTED_RUN_DIR = ""
FILENAME = ""
EVAL_JSON = ""
MANUAL_DOT_QUERIES_FILE = ""
CLUSTER_QUERY_IMAGE_PREFIX = ""
CLUSTER_QUERY_IMAGE_PATH = ""
BACKGROUND_REFERENCE_GEOJSON = ""
BASE_BACKGROUND_IMG = ""
SHIFTED_BACKGROUND_IMG = ""
CLUSTER_TYPE_GEOJSON = os.path.join(CLUSTER_TYPE_ROOT, f"{CLUSTER_TYPE}.geojson")

# Per-run file names for temp artifacts and outputs (allow parallel runs),
# all placed under RESULTS_RUN_DIR.
LEFT_GEOJSON = ""
RIGHT_GEOJSON = ""
UP_GEOJSON = ""
DOWN_GEOJSON = ""
NOCHANGE_GEOJSON = ""

LEFT_IMG = ""
RIGHT_IMG = ""
UP_IMG = ""
DOWN_IMG = ""
NOCHANGE_IMG = ""

BOUNDARY_PIXELS_IMG = ""

GEO_PLOT_FILE = ""
GEO_BOX_PDF_FILE = ""
DOT_RESULTS_FILE = ""
SUMMARY_RESULTS_FILE = ""
RUN_LOG_FILE = ""

max_lon = MIN_LON + PIXEL_SIZE * WIDTH_PX
min_lat = MAX_LAT - PIXEL_SIZE * HEIGHT_PX
dirs = [(-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),         ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)]

def configure_dataset(dataset_key):
    global CURRENT_DATASET_KEY, JSON_FILE, EVAL_SOURCE_FILE, RESULTS_RUN_DIR, AUGMENTED_RUN_DIR
    global FILENAME, EVAL_JSON, MANUAL_DOT_QUERIES_FILE, CLUSTER_QUERY_IMAGE_PREFIX, CLUSTER_QUERY_IMAGE_PATH
    global BACKGROUND_REFERENCE_GEOJSON, BASE_BACKGROUND_IMG, SHIFTED_BACKGROUND_IMG
    global LEFT_GEOJSON, RIGHT_GEOJSON, UP_GEOJSON, DOWN_GEOJSON, NOCHANGE_GEOJSON
    global LEFT_IMG, RIGHT_IMG, UP_IMG, DOWN_IMG, NOCHANGE_IMG
    global BOUNDARY_PIXELS_IMG, GEO_PLOT_FILE, GEO_BOX_PDF_FILE, DOT_RESULTS_FILE, SUMMARY_RESULTS_FILE, RUN_LOG_FILE

    CURRENT_DATASET_KEY = dataset_key
    JSON_FILE = resolve_input_path(RUN_DATASET[dataset_key])
    EVAL_SOURCE_FILE = resolve_input_path(EVAL_SOURCE_FILES.get(dataset_key, RUN_DATASET[dataset_key]))
    RESULTS_RUN_DIR = os.path.join(RESULTS_ROOT, TEST_NAME or "default", dataset_key)
    AUGMENTED_RUN_DIR = os.path.join(AUGMENTED_ROOT, TEST_NAME or "default", dataset_key)
    os.makedirs(RESULTS_RUN_DIR, exist_ok=True)
    os.makedirs(AUGMENTED_RUN_DIR, exist_ok=True)

    dataset_suffix = f"{RUN_SUFFIX}_{dataset_key}" if RUN_SUFFIX else f"_{dataset_key}"
    FILENAME = os.path.join(RESULTS_RUN_DIR, f"map{dataset_suffix}.png")
    EVAL_JSON = os.path.join(RESULTS_RUN_DIR, f"eval_{EVAL_DECIMALS}decimals{dataset_suffix}.geojson")
    MANUAL_DOT_QUERIES_FILE = os.path.join(AUGMENTED_RUN_DIR, f"manual_dot_queries{dataset_suffix}.txt")
    CLUSTER_QUERY_IMAGE_PREFIX = f"cluster_query_blob{dataset_suffix}"
    CLUSTER_QUERY_IMAGE_PATH = os.path.join(AUGMENTED_RUN_DIR, f"cluster_size{dataset_suffix}.png")
    BACKGROUND_REFERENCE_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"dummy_points{dataset_suffix}.geojson")
    BASE_BACKGROUND_IMG = os.path.join(AUGMENTED_RUN_DIR, f"background_base{dataset_suffix}.png")
    SHIFTED_BACKGROUND_IMG = os.path.join(AUGMENTED_RUN_DIR, f"background_shifted{dataset_suffix}.png")
    LEFT_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_left{dataset_suffix}.geojson")
    RIGHT_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_right{dataset_suffix}.geojson")
    UP_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_up{dataset_suffix}.geojson")
    DOWN_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_down{dataset_suffix}.geojson")
    NOCHANGE_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_no_change{dataset_suffix}.geojson")
    LEFT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"left_img{dataset_suffix}.png")
    RIGHT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"right_img{dataset_suffix}.png")
    UP_IMG = os.path.join(AUGMENTED_RUN_DIR, f"up_img{dataset_suffix}.png")
    DOWN_IMG = os.path.join(AUGMENTED_RUN_DIR, f"down_img{dataset_suffix}.png")
    NOCHANGE_IMG = os.path.join(AUGMENTED_RUN_DIR, f"no_change_img{dataset_suffix}.png")
    BOUNDARY_PIXELS_IMG = os.path.join(AUGMENTED_RUN_DIR, f"BoundaryPixels{dataset_suffix}.png")
    GEO_PLOT_FILE = os.path.join(RESULTS_RUN_DIR, f"geo_error_histogram_boxplot{dataset_suffix}.png")
    GEO_BOX_PDF_FILE = os.path.join(RESULTS_RUN_DIR, f"geo_error_boxplot{dataset_suffix}.pdf")
    DOT_RESULTS_FILE = os.path.join(RESULTS_RUN_DIR, f"dot_center_results{dataset_suffix}.txt")
    SUMMARY_RESULTS_FILE = os.path.join(RESULTS_RUN_DIR, f"summary_results{dataset_suffix}.txt")
    RUN_LOG_FILE = os.path.join(AUGMENTED_RUN_DIR, f"run_log{dataset_suffix}.txt")


# =========================
# File and Rendering Helpers
# =========================

def resolve_geojson_path(path: str) -> str:
    """
    Return a verified GeoJSON path or raise a friendly error with suggestions.
    """
    if os.path.exists(path):
        return path

    geojson_candidates = []
    for root, _, files in os.walk(BASE_DIR):
        for name in files:
            if name.lower().endswith(".geojson"):
                geojson_candidates.append(os.path.relpath(os.path.join(root, name), BASE_DIR))
    suggestions = get_close_matches(path, geojson_candidates, n=5, cutoff=0.5)
    suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise FileNotFoundError(f"GeoJSON file not found: {path}.{suggestion_text}")

def resolve_input_path(path: str) -> str:
    """
    Resolve a repo-relative dataset path to an absolute path.
    """
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(os.path.join(DATA_ROOT, path))
        candidates.append(os.path.join(BASE_DIR, path))

    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if os.path.exists(normalized):
            return normalized

    checked_paths = ", ".join(os.path.normpath(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Required input file not found: {path}. Checked: {checked_paths}")

def validate_required_inputs():
    missing = []
    for label, path in (
        ("JSON_FILE", JSON_FILE),
        ("EVAL_SOURCE_FILE", EVAL_SOURCE_FILE),
    ):
        if not os.path.exists(path):
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


configure_dataset(CURRENT_DATASET_KEY)

def safe_input():
    while True:
        line = input().strip()
        if line:
            return line

def create_geojson(latlon_list, output_path):
    """
    Creates a GeoJSON file from a list of (latitude, longitude) tuples.

    Parameters:
        latlon_list (list of (lat, lon)): List of geographic coordinates.
        output_path (str): Path to save the output .geojson file.
    """
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    for lat, lon in latlon_list:
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        }
        geojson_data["features"].append(feature)

    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)

def round_geometry_coordinates(coords, decimals):
    if isinstance(coords, list):
        if coords and isinstance(coords[0], (int, float)):
            return [round(value, decimals) if isinstance(value, (int, float)) else value for value in coords]
        return [round_geometry_coordinates(value, decimals) for value in coords]
    return coords

def prepare_eval_geojson(source_path=None, output_path=None, decimals=EVAL_DECIMALS):
    source_path = source_path or EVAL_SOURCE_FILE or JSON_FILE
    output_path = output_path or EVAL_JSON
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rounded = json.loads(json.dumps(data))
    for feature in rounded.get("features", []):
        geometry = feature.get("geometry")
        if geometry and "coordinates" in geometry:
            geometry["coordinates"] = round_geometry_coordinates(geometry["coordinates"], decimals)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rounded, f, indent=2)

    print(f"Prepared EVAL_JSON with {decimals} decimal places: {output_path}")

def write_dot_results_file(output_path, estimated_centers, width, height):
    """
    Write a compact per-dot results file with pixel and lat/lon centers.
    """
    with open(output_path, "w", encoding="utf-8") as rf:
        rf.write("DOT CENTER RESULTS\n")
        rf.write(f"TEST_NAME: {TEST_NAME}\n")
        rf.write(f"DATASET: {CURRENT_DATASET_KEY}\n")
        rf.write(f"INPUT_GEOJSON: {JSON_FILE}\n")
        rf.write(f"EVAL_GEOJSON: {EVAL_JSON}\n")
        rf.write(f"MAP_IMAGE: {FILENAME}\n")
        rf.write(f"AUGMENTED_DIR: {AUGMENTED_RUN_DIR}\n")
        rf.write(f"RUN_LOG: {RUN_LOG_FILE}\n")
        rf.write(f"POINT_COUNT: {len(estimated_centers)}\n\n")
        rf.write("index,pixel_x,pixel_y,latitude,longitude\n")
        for idx, center in enumerate(estimated_centers, start=1):
            lat, lon = pixel_to_lat_lon(center[0], center[1], width, height)
            rf.write(f"{idx},{center[0]:.6f},{center[1]:.6f},{lat:.8f},{lon:.8f}\n")

def get_true_points(filename=None):
    filename = filename or EVAL_JSON
    with open(filename, "r") as f:
        data = json.load(f)
    
    true_points = []
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords:
            true_points.append((coords[0], coords[1]))
    return true_points

def valid_latlon(latlon):
    if latlon is None or len(latlon) < 2:
        return False
    lat, lon = latlon[0], latlon[1]
    return (
        isinstance(lat, (int, float))
        and isinstance(lon, (int, float))
        and math.isfinite(lat)
        and math.isfinite(lon)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )

def filter_valid_latlon_pairs(true_latlons, pred_latlons, true_pixels=None, pred_pixels=None):
    filtered_true_latlons = []
    filtered_pred_latlons = []
    filtered_true_pixels = [] if true_pixels is not None else None
    filtered_pred_pixels = [] if pred_pixels is not None else None
    skipped = 0

    for idx, (true_latlon, pred_latlon) in enumerate(zip(true_latlons, pred_latlons)):
        if valid_latlon(true_latlon) and valid_latlon(pred_latlon):
            filtered_true_latlons.append(true_latlon)
            filtered_pred_latlons.append(pred_latlon)
            if filtered_true_pixels is not None:
                filtered_true_pixels.append(true_pixels[idx])
            if filtered_pred_pixels is not None:
                filtered_pred_pixels.append(pred_pixels[idx])
        else:
            skipped += 1

    return filtered_true_latlons, filtered_pred_latlons, filtered_true_pixels, filtered_pred_pixels, skipped

def lat_lon_to_pixel(lat, lon, img_width, img_height):
    x = (lon - MIN_LON) / (max_lon - MIN_LON) * img_width
    y = (MAX_LAT - lat) / (MAX_LAT - min_lat) * img_height
    return (x, y)

def pixel_to_lat_lon(x, y, img_width, img_height):
    lon = MIN_LON + (x / img_width) * (max_lon - MIN_LON)
    lat = MAX_LAT - (y / img_height) * (MAX_LAT - min_lat)
    return (lat, lon)


# =========================
# Math Helpers
# =========================

def color_distance(c1, c2):
    return (sum(abs(a - b) for a, b in zip(c1, c2)))

def mean_without_outliers(values, drop_fraction=0.2):
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    n = len(arr)
    drop = int(n * drop_fraction)
    if drop >= n:
        drop = n - 1
    kept = arr[: n - drop] if drop > 0 else arr
    if not kept:
        kept = arr
    return float(sum(kept) / len(kept))

def region_area(r, theta):
    mid_x = 0.5 + r * math.cos(theta)
    mid_y = 0.5 + r * math.sin(theta)
    if mid_x >= 1 or mid_y >= 1:
        return 1
    if mid_x <= 0 or mid_y <= 0:
        return 0
    if r < 0:
        if mid_x ** 2 + (1 - mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.cos(theta)
        elif (1 - mid_x) ** 2 + (mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.sin(theta)
        else:
            d = 1/math.sqrt(2) * math.cos(math.pi/4 - theta)
            return 1/2 * 1/math.cos(theta) * 1/math.sin(theta) * (d + r) ** 2
    else:
        if mid_x ** 2 + (1 - mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.sin(theta)
        elif (1 - mid_x) ** 2 + (mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.cos(theta)
        else:
            d = 1/math.sqrt(2) * math.cos(math.pi/4 - theta)
            return 1 - 1/2 * 1/math.cos(theta) * 1/math.sin(theta) * (d - r) ** 2

def find_r(theta, area):
    def objective(r):
        return region_area(r, theta) - area
    
    r_min = -10
    r_max = 10
    
    f_min = objective(r_min)
    f_max = objective(r_max)
    
    if f_min * f_max > 0:
        raise ValueError("No solution found within the range [-10, 10]. Ensure area is between 0 and 1 and theta is valid.")
    
    r_solution = brentq(objective, r_min, r_max)
    return r_solution

def estimate_radius_from_area(pixel_count, margin=1.5):
    return math.sqrt(max(1, pixel_count) / math.pi) + margin


# =========================
# GeoJSON and Evaluation Helpers
# =========================

def estimate_cluster_size(blob_pixel_count, min_blob_pixel_count):
    if min_blob_pixel_count <= 0:
        return 1

    cluster_ratio = blob_pixel_count / min_blob_pixel_count
    if cluster_ratio < 1.2:
        return 1

    estimated_size = 2
    while cluster_ratio >= estimated_size + 0.1:
        estimated_size += 1
    return estimated_size

def generate_unique_color(i):
    random.seed(i)
    return tuple(random.randint(50, 255) for _ in range(3))

def save_blob_query_image(image_copy, blob, blob_index, output_dir=None):
    """
    Save an image highlighting the given blob so manual cluster counting
    behaves like the original script.
    """
    vis = image_copy.copy()
    pix = vis.load()
    highlight = (0, 255, 255)
    for x, y in blob:
        pix[x, y] = highlight
    path = CLUSTER_QUERY_IMAGE_PATH
    vis.save(path)
    return path


# =========================
# Blob Detection Baselines
# =========================

def load_rgb_image_with_retry(image_path, retries=5, delay=0.2):
    last_error = None
    for attempt in range(retries):
        try:
            with Image.open(image_path) as img:
                return img.convert("RGB").copy()
        except OSError as exc:
            last_error = exc
            if "truncated" not in str(exc).lower() or attempt == retries - 1:
                raise
            time.sleep(delay)
    raise last_error

def generate_map(predicted_points: str, output_path: str, tile_source=None):
    """
    Generate a stretched visual map from a GeoJSON of predicted points
    and save it to the specified output path.
    """
    os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()

    predicted_points = resolve_geojson_path(predicted_points)
    gdf = gpd.read_file(predicted_points)
    gdf = gdf[
        gdf.geometry.notnull() &
        gdf.geometry.x.notnull() &
        gdf.geometry.y.notnull() &
        gdf.geometry.y.between(-89.999, 89.999)
    ]

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326)

    tile_source = tile_source or PRIMARY_TILE_SOURCE

    ctx.set_cache_dir(os.path.join(BASE_DIR, "osm_cache"))

    stretch_factor = 1
    fig_w_in = (WIDTH_PX / 96) * stretch_factor
    fig_h_in = HEIGHT_PX / 96
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=96)

    mm_to_pt = 72 / 25.4
    marker_diameter_mm = DOT_RADIUS_MM * 2
    marker_size_pts2 = (marker_diameter_mm * mm_to_pt) ** 2

    if not gdf.empty:
        gdf.plot(
            ax=ax,
            color="#ff0000",
            markersize=marker_size_pts2,
            edgecolor="none",
            linewidth=0,
            alpha=0 if "dummy_points.geojson" in predicted_points else 1
        )
    else:
        print(f"Warning: no valid points found in {predicted_points}; rendering basemap only.", flush=True)

    ctx.add_basemap(
        ax,
        source=tile_source,
        crs="EPSG:4326",
        zoom=MAP_ZOOM,
        reset_extent=False
    )

    ax.set_xlim(MIN_LON, max_lon)
    ax.set_ylim(min_lat, MAX_LAT)
    ax.set_aspect(1 / stretch_factor)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(output_path, dpi=96, bbox_inches=None, pad_inches=0, facecolor="none")
    plt.close(fig)
    print(f"Map saved to: {output_path}")

def ensure_background_reference_images():
    create_geojson([], BACKGROUND_REFERENCE_GEOJSON)
    generate_map(BACKGROUND_REFERENCE_GEOJSON, BASE_BACKGROUND_IMG, PRIMARY_TILE_SOURCE)
    generate_map(BACKGROUND_REFERENCE_GEOJSON, SHIFTED_BACKGROUND_IMG, SHIFTED_TILE_SOURCE)

def get_blobs(image):
    pixels = image.load()
    width, height = image.size
    visited = [[False for _ in range(height)] for _ in range(width)]
    blobs = []
    for x in range(width):
        for y in range(height):
            if pixels[x,y] == DOT_COLOR and not visited[x][y]:
                queue = deque()
                queue.append((x, y))
                blob = []
                while queue:
                    cx, cy = queue.popleft()
                    if not (0 <= cx < width and 0 <= cy < height):
                        continue
                    if visited[cx][cy] or pixels[cx, cy] != (255, 0, 0):
                        continue
                    visited[cx][cy] = True
                    blob.append((cx, cy))
                    for dx, dy in dirs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                            queue.append((nx, ny))
                blobs.append(blob)
    return blobs

def geometric_find_blob_centers(image_path, dot_color=DOT_COLOR):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = img.load()

    global_visited = set()
    predicted_centers = []

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def bfs(start_x, start_y):
        local_visited = set()
        cluster = []
        border_pixels = set()
        pixel_weight = defaultdict(lambda: 1.0)

        queue = deque([(start_x, start_y)])
        local_visited.add((start_x, start_y))
        cluster.append((start_x, start_y))

        while queue:
            a, b = queue.popleft()
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = a + dx, b + dy
                    if not in_bounds(nx, ny) or (nx, ny) in local_visited:
                        continue
                    local_visited.add((nx, ny))
                    if color_distance(pixels[nx, ny], dot_color) == 0:
                        queue.append((nx, ny))
                        cluster.append((nx, ny))
                    else:
                        cluster.append((nx, ny))
                        border_pixels.add((nx, ny))

        for a, b in border_pixels:
            neighborhood = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    na, nb = a + dx, b + dy
                    if in_bounds(na, nb) and (na, nb) not in local_visited and (na, nb) not in border_pixels:
                        neighborhood.append(pixels[na, nb])
            if not neighborhood:
                continue
            avg_bg = tuple(sum(p[i] for p in neighborhood) / len(neighborhood) for i in range(3))
            dist_to_bg = color_distance(avg_bg, pixels[a, b])
            dist_to_dot = color_distance(avg_bg, dot_color)
            pixel_weight[(a, b)] = min(1.0, dist_to_bg / max(1e-5, dist_to_dot))

        x_sum = y_sum = weight_sum = 0.0
        for x, y in cluster:
            w = pixel_weight[(x, y)]
            x_sum += w * x
            y_sum += w * y
            weight_sum += w

        if weight_sum < 1e-5:
            return None, None, []
        return x_sum / weight_sum + 0.5, y_sum / weight_sum + 0.5, cluster

    for x in range(width):
        for y in range(height):
            if (x, y) in global_visited or color_distance(pixels[x, y], dot_color) != 0:
                continue
            cx, cy, cluster_pixels = bfs(x, y)
            if cx is None or cy is None:
                continue
            predicted_centers.append((cx, cy))
            global_visited.update(cluster_pixels)

    return predicted_centers

def naive_find_blob_centers(image_path, dot_color=DOT_COLOR, include_border_pixels=True):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = img.load()
    visited = set()
    predicted_centers = []

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def bfs(start_x, start_y):
        cluster = []
        queue = deque([(start_x, start_y)])
        visited.add((start_x, start_y))
        cluster.append((start_x, start_y))
        x_sum = 0.0
        y_sum = 0.0
        weight_sum = 0.0

        while queue:
            a, b = queue.popleft()
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = a + dx, b + dy
                    if not in_bounds(nx, ny) or (nx, ny) in visited:
                        continue
                    if color_distance(pixels[nx, ny], dot_color) == 0:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        cluster.append((nx, ny))
                    elif include_border_pixels:
                        cluster.append((nx, ny))

        for x, y in cluster:
            x_sum += x
            y_sum += y
            weight_sum += 1.0

        return (x_sum / weight_sum + 0.5, y_sum / weight_sum + 0.5) if weight_sum else None

    for x in range(width):
        for y in range(height):
            if (x, y) in visited or color_distance(pixels[x, y], dot_color) != 0:
                continue
            dot_center = bfs(x, y)
            if dot_center is not None:
                predicted_centers.append(dot_center)

    return predicted_centers

def expand_centers_by_cluster_size(blob_centers, blob_sizes, label):
    if len(blob_centers) != len(blob_sizes):
        print(
            f"Warning: {label} found {len(blob_centers)} blob centers but expected {len(blob_sizes)} blobs; using first {min(len(blob_centers), len(blob_sizes))}.",
            flush=True,
        )

    expanded_centers = []
    expanded_sizes = []
    for center, cluster_size in zip(blob_centers, blob_sizes):
        expanded_centers.extend([center] * cluster_size)
        expanded_sizes.extend([cluster_size] * cluster_size)
    return expanded_centers, expanded_sizes

def match_estimates(true_centers, estimated_centers):
    """Greedy O(n^2) matching: repeatedly pick closest pair. Returns only matched pairs."""
    true_list = list(true_centers)
    est_list = list(estimated_centers)
    n_true = len(true_list)
    n_est = len(est_list)
    print(f"True points: {n_true}, Estimated points: {n_est}, Matched: {min(n_true, n_est)}")

    # Build all (dist_sq, i, j) and sort by distance
    pairs = []
    for i in range(n_true):
        for j in range(n_est):
            dx = true_list[i][0] - est_list[j][0]
            dy = true_list[i][1] - est_list[j][1]
            pairs.append((dx * dx + dy * dy, i, j))
    pairs.sort(key=lambda x: x[0])

    used_true = set()
    used_est = set()
    errors = []
    matched_true = []
    matched_est = []
    matched_true_indices = []
    matched_est_indices = []

    for _, i, j in pairs:
        if i in used_true or j in used_est:
            continue
        used_true.add(i)
        used_est.add(j)
        true_pt = true_list[i]
        est_pt = est_list[j]
        errors.append((abs(true_pt[0] - est_pt[0]), abs(true_pt[1] - est_pt[1])))
        matched_true.append(true_pt)
        matched_est.append(est_pt)
        matched_true_indices.append(i)
        matched_est_indices.append(j)

    return errors, matched_true, matched_est, matched_true_indices, matched_est_indices


# =========================
# Main Pipeline
# =========================

def main():
    """Run the original attack path without the old menu split."""
    validate_required_inputs()
    prepare_eval_geojson()
    if REGENERATE_BASE_MAP or not os.path.exists(FILENAME):
        generate_map(JSON_FILE, FILENAME, PRIMARY_TILE_SOURCE)

    if BG_MODE:
        ensure_background_reference_images()

    with open(MANUAL_DOT_QUERIES_FILE, "w", buffering=1) as mf:
        if CLUSTER_SIZE_MODE == "manual":
            mf.write(f"Manual dot count queries (stdin=terminal)\n{'=' * 50}\n")
        else:
            mf.write(f"Estimated cluster sizes\n{'=' * 50}\n")

    image = Image.open(FILENAME).convert("RGB")
    pixels = image.load()
    width, height = image.size
    if width != WIDTH_PX or height != HEIGHT_PX:
        print(f"WARNING: Original image size ({width}x{height}) doesn't match expected size ({WIDTH_PX}x{HEIGHT_PX})")
        print("This will cause coordinate misalignment. Please regenerate the map with option 1.")

    original_image = Image.open(FILENAME).convert("RGB")
    original_pixels = original_image.load()
    blobs = get_blobs(image)

    def fit_circle(xs, ys):
        x = np.array(xs)
        y = np.array(ys)
        design = np.c_[2 * x, 2 * y, np.ones_like(x)]
        target = x**2 + y**2
        coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        xc, yc, _ = coeffs
        return xc, yc

    def get_initial_centers():
        bfs_owner = [[-1 for _ in range(height)] for _ in range(width)]
        cluster_size_by_id = {}
        blob_sizes = []
        min_blob_pixels = min((len(blob) for blob in blobs), default=1)
        estimated_base_radius = estimate_radius_from_area(min_blob_pixels, margin=0.5)
        cluster_sizes = {}

        if CLUSTER_SIZE_MODE == "estimate":
            print(f"Minimum blob size (1-dot reference): {min_blob_pixels} red pixels", flush=True)
            print(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px", flush=True)
            with open(MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                mf.write(f"Minimum blob size (1-dot reference): {min_blob_pixels} red pixels\n")
                mf.write(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px\n")
        elif CLUSTER_SIZE_MODE == "manual":
            print(f"Minimum blob size (radius reference): {min_blob_pixels} red pixels", flush=True)
            print(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px", flush=True)
            with open(MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                mf.write(f"Minimum blob size (radius reference): {min_blob_pixels} red pixels\n")
                mf.write(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px\n")
            if os.path.exists(CLUSTER_TYPE_GEOJSON):
                with open(CLUSTER_TYPE_GEOJSON, "r", encoding="utf-8") as cf:
                    data = json.load(cf)
                for feat in data.get("features", []):
                    props = feat.get("properties", {})
                    key = props.get("key")
                    size = props.get("size")
                    if key is not None and size is not None:
                        cluster_sizes[str(key)] = int(size)
        else:
            raise ValueError(f"Unsupported CLUSTER_SIZE_MODE: {CLUSTER_SIZE_MODE}")

        for blob_index, blob in enumerate(blobs):
            print(f"Processing blob {blob_index + 1} with {len(blob)} red pixels")
            if CLUSTER_SIZE_MODE == "estimate":
                cluster_ratio = len(blob) / min_blob_pixels if min_blob_pixels else 1.0
                cluster_size = estimate_cluster_size(len(blob), min_blob_pixels)
                print(f"  Estimated {cluster_size} dots from ratio {cluster_ratio:.3f}", flush=True)
                with open(MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                    mf.write(f"Blob {blob_index + 1}: {len(blob)} red pixels, ratio = {cluster_ratio:.3f}, estimated dots = {cluster_size}\n")
            else:
                first_pixel = min(blob)
                key = f"{first_pixel[0]},{first_pixel[1]}"
                if key in cluster_sizes:
                    cluster_size = cluster_sizes[key]
                    print(f"  (cached: {cluster_size} dots from {CLUSTER_TYPE_GEOJSON})", flush=True)
                else:
                    img_path = save_blob_query_image(original_image, blob, blob_index)
                    with open(MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                        mf.write(f"Blob {blob_index + 1}: {len(blob)} red pixels. Image: {img_path}\n")
                        mf.flush()
                        prompt = f"Blob {blob_index + 1} ({len(blob)} red pixels): Enter number of dots in this cluster: "
                        print(prompt, file=sys.stderr)
                        mf.write(prompt)
                        mf.flush()
                    cluster_size = max(1, int(input().strip() or "1"))
                    with open(MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                        mf.write(f"{cluster_size}\n")
                        mf.flush()
                    cluster_sizes[key] = cluster_size

                    features = []
                    for key_str, size in cluster_sizes.items():
                        try:
                            x_str, y_str = key_str.split(",")
                            x0 = int(x_str)
                            y0 = int(y_str)
                        except Exception:
                            continue
                        features.append(
                            {
                                "type": "Feature",
                                "properties": {"key": key_str, "size": int(size)},
                                "geometry": {"type": "Point", "coordinates": [x0, y0]},
                            }
                        )
                    with open(CLUSTER_TYPE_GEOJSON, "w", encoding="utf-8") as cf:
                        json.dump({"type": "FeatureCollection", "features": features}, cf, indent=2)

            blob_sizes.append(cluster_size)
            centers = random.sample(blob, cluster_size)
            for _ in range(MAX_ITER):
                clusters = [[] for _ in range(cluster_size)]
                for x, y in blob:
                    distances = [math.hypot(cx - x, cy - y) for (cx, cy) in centers]
                    closest = distances.index(min(distances))
                    clusters[closest].append((x, y))
                new_centers = []
                for cluster in clusters:
                    if cluster:
                        avg_x = sum(p[0] + 0.5 for p in cluster) / len(cluster)
                        avg_y = sum(p[1] + 0.5 for p in cluster) / len(cluster)
                        new_centers.append((avg_x, avg_y))
                    else:
                        new_centers.append(random.choice(blob))
                if all(math.hypot(a - b, c - d) < TOL for (a, c), (b, d) in zip(centers, new_centers)):
                    break
                centers = new_centers

            for center_index, (center_x, center_y) in enumerate(centers):
                assigned_red_pixels = clusters[center_index] if center_index < len(clusters) else []
                cluster_search_radius = max(
                    estimated_base_radius,
                    estimate_radius_from_area(len(assigned_red_pixels), margin=0.5),
                )
                cx, cy = int(round(center_x)), int(round(center_y))
                queue = deque([(cx, cy)])
                seen = set()
                cluster_id = blob_index * 1000 + center_index
                cluster_size_by_id[cluster_id] = cluster_size

                while queue:
                    x, y = queue.popleft()
                    if not (0 <= x < width and 0 <= y < height):
                        continue
                    if (x, y) in seen:
                        continue
                    if math.hypot(x - center_x, y - center_y) > cluster_search_radius:
                        continue
                    seen.add((x, y))
                    if bfs_owner[x][y] != -1 and bfs_owner[x][y] != cluster_id and pixels[x, y] != DOT_COLOR:
                        continue
                    bfs_owner[x][y] = cluster_id
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) not in seen and pixels[x, y] == DOT_COLOR:
                            queue.append((nx, ny))

        to_unassign = set()
        for x in range(width):
            for y in range(height):
                cid = bfs_owner[x][y]
                if cid == -1:
                    continue
                is_border = False
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and bfs_owner[nx][ny] == -1:
                            is_border = True
                if not is_border:
                    continue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            neighbor_cid = bfs_owner[nx][ny]
                            if neighbor_cid != -1 and neighbor_cid != cid:
                                to_unassign.add((x, y))
                                to_unassign.add((nx, ny))

        for x, y in to_unassign:
            bfs_owner[x][y] = -1

        print(f"Unassigned {len(to_unassign)} border pixels due to king-adjacency conflicts.")

        for x in range(width):
            for y in range(height):
                if pixels[x, y] != DOT_COLOR and bfs_owner[x][y] != -1:
                    cluster_id = bfs_owner[x][y]
                    blob_index = cluster_id // 1000
                    center_index = cluster_id % 1000
                    pixels[x, y] = generate_unique_color(blob_index * 100 + center_index)

        image.save(BOUNDARY_PIXELS_IMG)
        print(f"Saved {BOUNDARY_PIXELS_IMG}")
        cluster_pixels = defaultdict(list)
        for x in range(width):
            for y in range(height):
                if pixels[x, y] != DOT_COLOR and bfs_owner[x][y] != -1:
                    cluster_pixels[bfs_owner[x][y]].append((x, y))

        circle_centers = {}
        for cid, pixels_list in cluster_pixels.items():
            xs = [x for x, y in pixels_list]
            ys = [y for x, y in pixels_list]
            circle_centers[cid] = fit_circle(xs, ys) if len(xs) >= 3 else (None, None)

        refined_centers = []
        for cid, (xc, yc) in circle_centers.items():
            if xc is None or yc is None:
                refined_centers.append((cid, (None, None)))
                continue
            adjusted_points = []
            xc_shifted = xc + 0.5
            yc_shifted = yc + 0.5
            for x, y in cluster_pixels[cid]:
                x_shifted = x + 0.5
                y_shifted = y + 0.5
                px_val = original_pixels[int(x), int(y)]
                denom = color_distance(BACKGROUND_COLOR, DOT_COLOR)
                if denom == 0:
                    denom = 1e-6
                ratio = color_distance(px_val, BACKGROUND_COLOR) / denom
                ratio = max(min(ratio, 1), 0.01)
                theta = math.atan2(abs(y_shifted - yc_shifted), abs(x_shifted - xc_shifted))
                signed_theta = math.atan2(y_shifted - yc_shifted, x_shifted - xc_shifted)
                try:
                    r_offset = find_r(theta, ratio)
                except Exception:
                    r_offset = 0.0
                if r_offset > 5:
                    r_offset = 5
                elif r_offset < -5:
                    r_offset = -5
                radius = math.hypot(abs(x_shifted - xc_shifted), abs(y_shifted - yc_shifted)) + r_offset
                adjusted_points.append(
                    (
                        xc_shifted + radius * math.cos(signed_theta),
                        yc_shifted + radius * math.sin(signed_theta),
                    )
                )
            if len(adjusted_points) >= 3:
                refined_centers.append((cid, fit_circle([x for x, y in adjusted_points], [y for x, y in adjusted_points])))
            else:
                refined_centers.append((cid, (None, None)))
        return refined_centers, cluster_pixels, cluster_size_by_id, blob_sizes

    initial_centers, cluster_pixels, cluster_size_by_id, blob_sizes = get_initial_centers()
    valid_centers = [(cid, center) for cid, center in initial_centers if center[0] is not None and center[1] is not None]
    pairs = sorted(valid_centers, key=lambda item: (item[1][0], item[1][1]))
    prev_centers = [center for _, center in pairs]
    keys = [cid for cid, _ in pairs]
    modified_cluster_sizes = [cluster_size_by_id[cid] for cid in keys]

    base_background_pixels = None
    shifted_background_pixels = None
    if BG_MODE:
        base_background_pixels = Image.open(BASE_BACKGROUND_IMG).convert("RGB").load()
        shifted_background_pixels = Image.open(SHIFTED_BACKGROUND_IMG).convert("RGB").load()

    def calc_error_for_dot(base, new):
        king_dirs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        scores = [0.0 for _ in range(len(keys))]
        for i in range(len(keys)):
            error = 0.0
            for x, y in cluster_pixels[keys[i]]:
                neighbor_pixels = []
                for dx, dy in king_dirs:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    has_dot_neighbor = False
                    for kdx, kdy in king_dirs:
                        kx, ky = nx + kdx, ny + kdy
                        if 0 <= kx < width and 0 <= ky < height and tuple(new[kx, ky]) == DOT_COLOR:
                            has_dot_neighbor = True
                            break
                    if not has_dot_neighbor:
                        neighbor_pixels.append((nx, ny))
                if BG_MODE:
                    new_background = shifted_background_pixels[x, y]
                    base_background = base_background_pixels[x, y]
                else:
                    new_background = BACKGROUND_COLOR
                    base_background = BACKGROUND_COLOR
                    if neighbor_pixels:
                        r = g = b = 0
                        for nx, ny in neighbor_pixels:
                            cr, cg, cb = new[nx, ny]
                            r += cr
                            g += cg
                            b += cb
                        count = len(neighbor_pixels)
                        new_background = (r // count, g // count, b // count)
                        r = g = b = 0
                        for nx, ny in neighbor_pixels:
                            cr, cg, cb = base[nx, ny]
                            r += cr
                            g += cg
                            b += cb
                        base_background = (r // count, g // count, b // count)
                d_new_bg_dot = color_distance(new_background, DOT_COLOR)
                d_base_bg_dot = color_distance(base_background, DOT_COLOR)
                if d_new_bg_dot == 0 or d_base_bg_dot == 0:
                    continue
                new_term = color_distance(new[x, y], new_background) / d_new_bg_dot
                base_term = color_distance(base[x, y], base_background) / d_base_bg_dot
                error += abs(new_term - base_term)
            scores[i] = error
        return scores

    def print_iteration_metrics():
        estimated_centers = prev_centers
        true_locations = get_true_points()
        true_centers = [lat_lon_to_pixel(loc[1], loc[0], width, height) for loc in true_locations]
        _, matched_true_centers, matched_est_centers, _, _ = match_estimates(true_centers, estimated_centers)
        if not matched_true_centers:
            print("\nNo matched pairs - skipping error metrics", flush=True)
            return
        matched_true_latlons = [pixel_to_lat_lon(t[0], t[1], width, height) for t in matched_true_centers]
        matched_pred_latlons = [pixel_to_lat_lon(e[0], e[1], width, height) for e in matched_est_centers]
        matched_true_latlons, matched_pred_latlons, matched_true_centers, matched_est_centers, skipped_invalid = filter_valid_latlon_pairs(
            matched_true_latlons,
            matched_pred_latlons,
            matched_true_centers,
            matched_est_centers,
        )
        if skipped_invalid:
            print(f"Warning: skipped {skipped_invalid} invalid lat/lon pairs during iteration metrics.", flush=True)
        if not matched_true_latlons:
            print("\nNo valid matched lat/lon pairs - skipping error metrics", flush=True)
            return
        geo_errors = [geodesic(true, pred).meters for true, pred in zip(matched_true_latlons, matched_pred_latlons)]
        x_errors = [abs(pred[0] - true[0]) for true, pred in zip(matched_true_centers, matched_est_centers)]
        y_errors = [abs(pred[1] - true[1]) for true, pred in zip(matched_true_centers, matched_est_centers)]
        lat_errors = [abs(pred[0] - true[0]) for true, pred in zip(matched_true_latlons, matched_pred_latlons)]
        lon_errors = [abs(pred[1] - true[1]) for true, pred in zip(matched_true_latlons, matched_pred_latlons)]
        p25, p50, p75 = [float(np.percentile(geo_errors, q)) for q in (25, 50, 75)]
        print(f"\nIteration avg pixel error (without outliers): x = {mean_without_outliers(x_errors):.8f} px, y = {mean_without_outliers(y_errors):.8f} px", flush=True)
        print(f"Iteration avg pixel error (with outliers): x = {(sum(x_errors) / len(x_errors) if x_errors else 0.0):.8f} px, y = {(sum(y_errors) / len(y_errors) if y_errors else 0.0):.8f} px", flush=True)
        print(f"Iteration median pixel error: x = {(float(np.median(x_errors)) if x_errors else 0.0):.8f} px, y = {(float(np.median(y_errors)) if y_errors else 0.0):.8f} px", flush=True)
        print(f"Iteration avg geodesic error (without outliers): {mean_without_outliers(geo_errors):.2f} meters", flush=True)
        print(f"Iteration avg geodesic error (with outliers): {(sum(geo_errors) / len(geo_errors) if geo_errors else 0.0):.2f} meters", flush=True)
        print(f"Iteration geodesic percentiles: 25th = {p25:.2f} m, 50th = {p50:.2f} m, 75th = {p75:.2f} m", flush=True)
        print(f"Iteration avg lat error (without outliers): {mean_without_outliers(lat_errors):.6f} deg, lon error: {mean_without_outliers(lon_errors):.6f} deg", flush=True)
        print(f"Iteration avg lat error (with outliers): {(sum(lat_errors) / len(lat_errors) if lat_errors else 0.0):.6f} deg, lon error: {(sum(lon_errors) / len(lon_errors) if lon_errors else 0.0):.6f} deg", flush=True)
        print(f"Iteration median lat error: {(float(np.median(lat_errors)) if lat_errors else 0.0):.6f} deg, lon error: {(float(np.median(lon_errors)) if lon_errors else 0.0):.6f} deg", flush=True)

    step_size = INITIAL_STEP_SIZE
    while step_size > MIN_STEP_SIZE:
        left = [(prev_centers[i][0] - step_size, prev_centers[i][1]) for i in range(len(keys))]
        left_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in left]
        create_geojson(left_coords, LEFT_GEOJSON)
        generate_map(LEFT_GEOJSON, LEFT_IMG, SHIFTED_TILE_SOURCE)
        left_err = calc_error_for_dot(original_pixels, load_rgb_image_with_retry(LEFT_IMG).load())

        right = [(prev_centers[i][0] + step_size, prev_centers[i][1]) for i in range(len(keys))]
        right_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in right]
        create_geojson(right_coords, RIGHT_GEOJSON)
        generate_map(RIGHT_GEOJSON, RIGHT_IMG, SHIFTED_TILE_SOURCE)
        right_err = calc_error_for_dot(original_pixels, load_rgb_image_with_retry(RIGHT_IMG).load())

        up = [(prev_centers[i][0], prev_centers[i][1] + step_size) for i in range(len(keys))]
        up_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in up]
        create_geojson(up_coords, UP_GEOJSON)
        generate_map(UP_GEOJSON, UP_IMG, SHIFTED_TILE_SOURCE)
        up_err = calc_error_for_dot(original_pixels, load_rgb_image_with_retry(UP_IMG).load())

        down = [(prev_centers[i][0], prev_centers[i][1] - step_size) for i in range(len(keys))]
        down_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in down]
        create_geojson(down_coords, DOWN_GEOJSON)
        generate_map(DOWN_GEOJSON, DOWN_IMG, SHIFTED_TILE_SOURCE)
        down_err = calc_error_for_dot(original_pixels, load_rgb_image_with_retry(DOWN_IMG).load())

        no_change = [(prev_centers[i][0], prev_centers[i][1]) for i in range(len(keys))]
        no_change_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in no_change]
        create_geojson(no_change_coords, NOCHANGE_GEOJSON)
        generate_map(NOCHANGE_GEOJSON, NOCHANGE_IMG, SHIFTED_TILE_SOURCE)
        no_change_err = calc_error_for_dot(original_pixels, load_rgb_image_with_retry(NOCHANGE_IMG).load())

        updated_centers = [(0, 0) for _ in range(len(keys))]
        for i in range(len(keys)):
            min_err = min(left_err[i], right_err[i], up_err[i], down_err[i], no_change_err[i])
            if left_err[i] == min_err:
                updated_centers[i] = left[i]
            elif right_err[i] == min_err:
                updated_centers[i] = right[i]
            elif up_err[i] == min_err:
                updated_centers[i] = up[i]
            elif down_err[i] == min_err:
                updated_centers[i] = down[i]
            else:
                updated_centers[i] = no_change[i]
        prev_centers = updated_centers
        print_iteration_metrics()
        step_size /= STEP_DIVISOR

    true_locations = get_true_points()
    true_centers = [lat_lon_to_pixel(loc[1], loc[0], width, height) for loc in true_locations]

    def build_method_summary(label, estimated_centers, est_cluster_sizes):
        _, matched_true_centers, matched_est_centers, _, matched_est_indices = match_estimates(true_centers, estimated_centers)
        if not matched_true_centers:
            return [label, "  No matched pairs."], None
        matched_true_latlons = [pixel_to_lat_lon(t[0], t[1], width, height) for t in matched_true_centers]
        matched_pred_latlons = [pixel_to_lat_lon(e[0], e[1], width, height) for e in matched_est_centers]
        matched_true_latlons, matched_pred_latlons, matched_true_centers, matched_est_centers, skipped_invalid = filter_valid_latlon_pairs(
            matched_true_latlons,
            matched_pred_latlons,
            matched_true_centers,
            matched_est_centers,
        )
        if not matched_true_latlons:
            return [label, "  No valid matched lat/lon pairs."], None
        geo_errors = [geodesic(true, pred).meters for true, pred in zip(matched_true_latlons, matched_pred_latlons)]
        x_errors = [abs(pred[0] - true[0]) for true, pred in zip(matched_true_centers, matched_est_centers)]
        y_errors = [abs(pred[1] - true[1]) for true, pred in zip(matched_true_centers, matched_est_centers)]
        lat_errors = [abs(pred[0] - true[0]) for true, pred in zip(matched_true_latlons, matched_pred_latlons)]
        lon_errors = [abs(pred[1] - true[1]) for true, pred in zip(matched_true_latlons, matched_pred_latlons)]

        cluster_size_geo_errors = defaultdict(list)
        for est_idx, geo_error in zip(matched_est_indices, geo_errors):
            if est_idx < len(est_cluster_sizes):
                cluster_size_geo_errors[est_cluster_sizes[est_idx]].append(geo_error)

        percentiles = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
        geo_arr = np.array(geo_errors)
        lines = [
            label,
            f"  Skipped invalid lat/lon pairs: {skipped_invalid}",
            "  WITHOUT OUTLIERS (best 80%):",
            f"    Avg pixel error: x = {mean_without_outliers(x_errors, drop_fraction=0.2):.6f} px, y = {mean_without_outliers(y_errors, drop_fraction=0.2):.6f} px",
            f"    Avg geodesic error: {mean_without_outliers(geo_errors, drop_fraction=0.2):.2f} m",
            f"    Avg lat error: {mean_without_outliers(lat_errors, drop_fraction=0.2):.6f} deg",
            f"    Avg lon error: {mean_without_outliers(lon_errors, drop_fraction=0.2):.6f} deg",
            "  WITH OUTLIERS (all):",
            f"    Avg pixel error: x = {(sum(x_errors) / len(x_errors) if x_errors else 0.0):.6f} px, y = {(sum(y_errors) / len(y_errors) if y_errors else 0.0):.6f} px",
            f"    Median pixel error: x = {(float(np.median(x_errors)) if x_errors else 0.0):.6f} px, y = {(float(np.median(y_errors)) if y_errors else 0.0):.6f} px",
            f"    Avg geodesic error: {(sum(geo_errors) / len(geo_errors) if geo_errors else 0.0):.2f} m",
            f"    Avg lat error: {(sum(lat_errors) / len(lat_errors) if lat_errors else 0.0):.6f} deg",
            f"    Avg lon error: {(sum(lon_errors) / len(lon_errors) if lon_errors else 0.0):.6f} deg",
            f"    Median lat error: {(float(np.median(lat_errors)) if lat_errors else 0.0):.6f} deg",
            f"    Median lon error: {(float(np.median(lon_errors)) if lon_errors else 0.0):.6f} deg",
            "  GEODESIC ERROR PERCENTILES (meters):",
        ]
        for q in percentiles:
            lines.append(f"    {q:3d}th: {float(np.percentile(geo_arr, q)):.2f} m")

        lines.append("  GEODESIC ERROR BY CLUSTER SIZE (meters):")
        for cluster_size in sorted(cluster_size_geo_errors):
            values = cluster_size_geo_errors[cluster_size]
            avg_val = float(sum(values) / len(values))
            median_val = float(np.median(values))
            lines.append(f"    Size {cluster_size}: avg = {avg_val:.2f} m, median = {median_val:.2f} m, n = {len(values)}")

        return lines, {"geo_errors": geo_errors}

    png_path = FILENAME
    modified_lines, modified_metrics = build_method_summary("Modified Method", prev_centers, modified_cluster_sizes)
    geometric_blob_centers = geometric_find_blob_centers(png_path)
    geometric_centers, geometric_cluster_sizes = expand_centers_by_cluster_size(geometric_blob_centers, blob_sizes, "Geometric")
    geometric_lines, _ = build_method_summary("Geometric", geometric_centers, geometric_cluster_sizes)
    pixelmatch_blob_centers = naive_find_blob_centers(png_path, include_border_pixels=True)
    pixelmatch_centers, pixelmatch_cluster_sizes = expand_centers_by_cluster_size(pixelmatch_blob_centers, blob_sizes, "PixelMatch")
    pixelmatch_lines, _ = build_method_summary("PixelMatch", pixelmatch_centers, pixelmatch_cluster_sizes)
    pixelavg_blob_centers = naive_find_blob_centers(png_path, include_border_pixels=False)
    pixelavg_centers, pixelavg_cluster_sizes = expand_centers_by_cluster_size(pixelavg_blob_centers, blob_sizes, "PixelAvg")
    pixelavg_lines, _ = build_method_summary("PixelAvg", pixelavg_centers, pixelavg_cluster_sizes)

    print("\n" + "=" * 60 + "\nFINAL RESULTS\n" + "=" * 60, flush=True)
    for block in (modified_lines, geometric_lines, pixelmatch_lines, pixelavg_lines):
        print("", flush=True)
        for line in block:
            print(line, flush=True)

    try:
        write_dot_results_file(DOT_RESULTS_FILE, prev_centers, width, height)
    except Exception as exc:
        print(f"Warning: could not write dot results file {DOT_RESULTS_FILE}: {exc}", flush=True)

    try:
        with open(SUMMARY_RESULTS_FILE, "w", encoding="utf-8") as sf:
            sf.write("FINAL SUMMARY RESULTS\n")
            sf.write(f"TEST_NAME: {TEST_NAME}\n")
            sf.write(f"CLUSTER_TYPE: {CLUSTER_TYPE}\n")
            sf.write(f"EVAL_JSON: {EVAL_JSON}\n")
            sf.write(f"EVAL_DECIMALS: {EVAL_DECIMALS}\n\n")
            for block in (modified_lines, geometric_lines, pixelmatch_lines, pixelavg_lines):
                for line in block:
                    sf.write(line + "\n")
                sf.write("\n")
    except Exception as exc:
        print(f"Warning: could not write summary file {SUMMARY_RESULTS_FILE}: {exc}", flush=True)

    if modified_metrics is None:
        print("\nNo matched pairs - cannot compute final metrics", flush=True)
        return

    good_geo_errors = modified_metrics["geo_errors"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(good_geo_errors, bins=max(1, min(30, len(good_geo_errors))), edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Geodesic error (m)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram of geodesic errors")
    ax2.boxplot(good_geo_errors, vert=True)
    ax2.set_ylabel("Geodesic error (m)")
    ax2.set_title("Box plot of geodesic errors")
    plt.tight_layout()
    plt.savefig(GEO_PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax_box = plt.subplots(1, 1, figsize=(4, 6))
    ax_box.boxplot(good_geo_errors, vert=True)
    ax_box.set_ylabel("Geodesic error (m)")
    ax_box.set_title("Box plot of geodesic errors")
    plt.tight_layout()
    plt.savefig(GEO_BOX_PDF_FILE, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved: {GEO_PLOT_FILE} and {GEO_BOX_PDF_FILE}", flush=True)
    print("=" * 60, flush=True)

class FlushingFile:
    """File wrapper that flushes after each write."""

    def __init__(self, file):
        self.file = file

    def write(self, text):
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.file.flush()


class TeeOutput:
    """Mirror stdout to both the terminal and the per-run log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    for dataset_key in RUN_DATASET:
        configure_dataset(dataset_key)
        with open(RUN_LOG_FILE, "w", buffering=1) as f:
            tee_output = TeeOutput(os.sys.__stdout__, FlushingFile(f))
            with contextlib.redirect_stdout(tee_output):
                print(f"DATASET: {CURRENT_DATASET_KEY}", flush=True)
                print(f"JSON_FILE: {JSON_FILE}", flush=True)
                print(f"EVAL_JSON: {EVAL_JSON}", flush=True)
                print(f"RESULTS_RUN_DIR: {RESULTS_RUN_DIR}", flush=True)
                print(f"AUGMENTED_RUN_DIR: {AUGMENTED_RUN_DIR}", flush=True)
                print(f"RUN_LOG_FILE: {RUN_LOG_FILE}", flush=True)
                main()

from PIL import Image
from collections import deque, defaultdict
import random
import math
from scipy.optimize import brentq
import json
import numpy as np
import os
import shutil
import sys
import xyzservices.providers as xyz
from . import config as cfg
from . import utils

"""
Recover point locations from a rendered dot map with perceptual descent.

Usage:
1. Run `python -m attack --input-image <map.png> --output <locations.geojson> ...`.
2. Read the recovered Point GeoJSON. Evaluation is a separate workflow.

 Dataset options:
- `OpenAddresses` uses `CoordinateJSONs/OpenAddress/US.geojson`
- `Synthetic` uses `CoordinateJSONs/Synthetic/US.geojson`
"""


# =========================
# Runtime Config
# =========================
def configure_cli(argv=None):
    args = cfg.parse_args(argv)
    cfg.apply_args(args)
    cfg.PRIMARY_TILE_SOURCE = xyz.OpenStreetMap.Mapnik
    cfg.SHIFTED_TILE_SOURCE = cfg.PRIMARY_TILE_SOURCE if cfg.BG_MODE else xyz.CartoDB.PositronNoLabels
    cfg.BACKGROUND_COLOR = (255, 255, 255)


def log(message="", *, verbose=False):
    if not cfg.QUIET and (not verbose or cfg.LOG_LEVEL == "verbose"):
        print(message, flush=True)


def is_dot_pixel(pixel, dot_color=cfg.DOT_COLOR):
    return utils.is_dot_pixel(pixel, dot_color=dot_color)


def is_solid_dot_pixel(pixel, dot_color=cfg.DOT_COLOR):
    return utils.is_solid_dot_pixel(pixel, dot_color=dot_color)


def apply_pixel_offset_to_centers(centers, offset_x=0.0, offset_y=0.0, width=None, height=None):
    shifted_centers = []
    for x, y in centers:
        shifted_x = x + offset_x
        shifted_y = y + offset_y
        if width is not None:
            shifted_x = min(max(shifted_x, 0.0), max(0.0, width - 1))
        if height is not None:
            shifted_y = min(max(shifted_y, 0.0), max(0.0, height - 1))
        shifted_centers.append((shifted_x, shifted_y))
    return shifted_centers


def trimmed_mean(values, drop_fraction=0.2):
    if not values:
        return 0.0
    ordered = sorted(values)
    drop_count = min(int(len(ordered) * drop_fraction), len(ordered) - 1)
    kept = ordered[: len(ordered) - drop_count]
    return sum(kept) / len(kept)


def estimate_triangle_anchor_offset(image_path, width, height):
    """Legacy truth-based calibration retained for compatibility diagnostics."""
    true_locations = utils.load_ground_truth_points()
    true_centers = [utils.geographic_to_pixel(loc[1], loc[0], width, height) for loc in true_locations]
    estimated_centers = geometric_component_center_estimates(image_path, refine_circles=False)
    _, matched_true, matched_est, _, _ = greedily_match_centers(true_centers, estimated_centers)
    if not matched_true:
        return (0.0, 0.0)
    return (
        trimmed_mean([true[0] - est[0] for true, est in zip(matched_true, matched_est)]),
        trimmed_mean([true[1] - est[1] for true, est in zip(matched_true, matched_est)]),
    )


def l1_color_distance(c1, c2):
    return (sum(abs(a - b) for a, b in zip(c1, c2)))

def unit_square_region_area(r, theta):
    theta = min(max(theta, 1e-6), math.pi / 2 - 1e-6)
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

def solve_radius_for_unit_square_area(theta, area):
    def objective(r):
        return unit_square_region_area(r, theta) - area
    
    r_min = -10
    r_max = 10
    
    f_min = objective(r_min)
    f_max = objective(r_max)
    
    if f_min * f_max > 0:
        raise ValueError("No solution found within the range [-10, 10]. Ensure area is between 0 and 1 and theta is valid.")
    
    r_solution = brentq(objective, r_min, r_max)
    return r_solution

def estimate_blob_radius_from_pixel_count(pixel_count, margin=1.5):
    return math.sqrt(max(1, pixel_count) / math.pi) + margin


def is_circle_like_dot_shape(dot_shape=None):
    dot_shape = cfg.DOT_SHAPE if dot_shape is None else dot_shape
    if dot_shape is None:
        return True
    if isinstance(dot_shape, tuple) and dot_shape:
        try:
            return int(dot_shape[0]) >= 5
        except (TypeError, ValueError):
            return False
    return False


# =========================
# GeoJSON and Evaluation Helpers
# =========================

def estimate_cluster_count_from_blob_pixels(blob_pixel_count, min_blob_pixel_count):
    if min_blob_pixel_count <= 0:
        return 1

    cluster_ratio = blob_pixel_count / min_blob_pixel_count
    # Single markers vary noticeably by shape and antialiasing.
    # Real connected 2-dot blobs are close to 2x; keep the single-dot band generous.
    if cluster_ratio < 1.5:
        return 1

    estimated_size = 2
    while cluster_ratio >= estimated_size + 0.1:
        estimated_size += 1
    return estimated_size

def generate_cluster_fill_color(i):
    random.seed(i)
    return tuple(random.randint(50, 255) for _ in range(3))

def save_cluster_query_preview(image_copy, blob, blob_index, output_dir=None):
    """
    Save an image highlighting the given blob so manual cluster counting
    behaves like the original script.
    """
    vis = image_copy.copy()
    pix = vis.load()
    highlight = (0, 255, 255)
    for x, y in blob:
        pix[x, y] = highlight
    path = cfg.CLUSTER_QUERY_IMAGE_PATH
    utils.ensure_parent_dir(path)
    vis.save(path)
    return path


# =========================
# Blob Detection Baselines
# =========================

def collect_exact_dot_components(image):
    pixels = image.load()
    width, height = image.size
    visited = [[False for _ in range(height)] for _ in range(width)]
    components = []
    for x in range(width):
        for y in range(height):
            if is_solid_dot_pixel(pixels[x, y]) and not visited[x][y]:
                queue = deque()
                queue.append((x, y))
                component = []
                while queue:
                    cx, cy = queue.popleft()
                    if not (0 <= cx < width and 0 <= cy < height):
                        continue
                    if visited[cx][cy] or not is_solid_dot_pixel(pixels[cx, cy]):
                        continue
                    visited[cx][cy] = True
                    component.append((cx, cy))
                    for dx, dy in cfg.DIRS:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                            queue.append((nx, ny))
                components.append(component)
    return components


def expected_dot_component_min_pixels():
    radius_px = cfg.DOT_RADIUS_MM * 96 / 25.4
    return max(3, int(math.pi * radius_px * radius_px * 0.08))


def is_plausible_compressed_dot_component(component):
    if len(component) < expected_dot_component_min_pixels():
        return False

    xs = [x for x, _ in component]
    ys = [y for _, y in component]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    skinny_ratio = max(width, height) / max(1, min(width, height))
    return skinny_ratio <= 8.0


def collect_compressed_dot_components(image):
    pixels = image.load()
    width, height = image.size
    visited = [[False for _ in range(height)] for _ in range(width)]
    components = []
    for x in range(width):
        for y in range(height):
            if visited[x][y] or not is_solid_dot_pixel(pixels[x, y]):
                continue
            queue = deque([(x, y)])
            component = []
            while queue:
                cx, cy = queue.popleft()
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                if visited[cx][cy] or not is_solid_dot_pixel(pixels[cx, cy]):
                    continue
                visited[cx][cy] = True
                component.append((cx, cy))
                for dx, dy in cfg.DIRS:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                        queue.append((nx, ny))
            if is_plausible_compressed_dot_component(component):
                components.append(component)
    return components


def collect_dot_components(image):
    if getattr(cfg, "IMAGE_FORMAT", "png") == "jpeg":
        return collect_compressed_dot_components(image)
    return collect_exact_dot_components(image)


def average_rgb(colors):
    count = len(colors)
    if count == 0:
        return None
    return tuple(sum(color[i] for color in colors) / count for i in range(3))


def geometric_component_center_estimates(
    image_path,
    dot_color=cfg.DOT_COLOR,
    background_image_path=None,
    refine_circles=None,
    refine_component_indices=None,
):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = img.load()
    background_image = None
    background_pixels = None
    if background_image_path and os.path.exists(background_image_path):
        background_image = Image.open(background_image_path).convert("RGB")
        if background_image.size == img.size:
            background_pixels = background_image.load()
        else:
            log(
                f"Warning: background image size {background_image.size} does not match map size {img.size}; using local background estimates.",
                verbose=True,
            )
    if refine_circles is None:
        refine_circles = is_circle_like_dot_shape()

    global_visited = set()
    dots_info = []
    dot_radii = []

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def boundary_background_pixel(x, y, local_visited, border_pixels):
        if background_pixels is not None:
            return background_pixels[x, y]
        neighborhood = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny) and (nx, ny) not in local_visited and (nx, ny) not in border_pixels:
                    neighborhood.append(pixels[nx, ny])
        return average_rgb(neighborhood)

    def alpha_weight_from_background(pixel, background_pixel):
        if background_pixel is None:
            return None
        denom = l1_color_distance(background_pixel, dot_color)
        if denom <= 1e-5:
            return None
        return max(0.0, min(1.0, l1_color_distance(pixel, background_pixel) / denom))

    def first_bfs(start_x, start_y):
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
                    if is_solid_dot_pixel(pixels[nx, ny], dot_color=dot_color):
                        queue.append((nx, ny))
                        cluster.append((nx, ny))
                    else:
                        cluster.append((nx, ny))
                        border_pixels.add((nx, ny))

        for a, b in border_pixels:
            background_pixel = boundary_background_pixel(a, b, local_visited, border_pixels)
            alpha_weight = alpha_weight_from_background(pixels[a, b], background_pixel)
            if alpha_weight is None:
                continue
            pixel_weight[(a, b)] = alpha_weight

        x_sum = y_sum = weight_sum = 0.0
        for x, y in cluster:
            w = pixel_weight[(x, y)]
            x_sum += w * x
            y_sum += w * y
            weight_sum += w

        if weight_sum < 1e-5:
            return None, None, None, None
        center_x = x_sum / weight_sum + 0.5
        center_y = y_sum / weight_sum + 0.5
        radius = math.sqrt(weight_sum / math.pi)
        return center_x, center_y, radius, cluster

    def second_bfs(start_x, start_y, approx_x, approx_y, approx_radius, threshold):
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
                    if is_solid_dot_pixel(pixels[nx, ny], dot_color=dot_color):
                        queue.append((nx, ny))
                        cluster.append((nx, ny))
                    else:
                        cluster.append((nx, ny))
                        border_pixels.add((nx, ny))

        for a, b in border_pixels:
            dx = abs((a + 0.5) - approx_x)
            dy = abs((b + 0.5) - approx_y)
            distance = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            geometry_weight = unit_square_region_area(approx_radius - distance, theta)

            background_pixel = boundary_background_pixel(a, b, local_visited, border_pixels)
            alpha_weight = alpha_weight_from_background(pixels[a, b], background_pixel)
            if alpha_weight is None or abs(alpha_weight - geometry_weight) > threshold:
                pixel_weight[(a, b)] = geometry_weight
            else:
                pixel_weight[(a, b)] = alpha_weight

        x_sum = y_sum = weight_sum = 0.0
        for x, y in cluster:
            w = pixel_weight[(x, y)]
            x_sum += w * x
            y_sum += w * y
            weight_sum += w

        if weight_sum < 1e-5:
            return None
        return x_sum / weight_sum + 0.5, y_sum / weight_sum + 0.5

    for x in range(width):
        for y in range(height):
            if (x, y) in global_visited or not is_solid_dot_pixel(pixels[x, y], dot_color=dot_color):
                continue
            cx, cy, radius, cluster_pixels = first_bfs(x, y)
            if cx is None or cy is None:
                continue
            dots_info.append((cx, cy, radius, cluster_pixels))
            dot_radii.append(radius)
            global_visited.update(cluster_pixels)

    if not refine_circles or not dot_radii:
        return [(cx, cy) for cx, cy, _, _ in dots_info]

    if refine_component_indices is None:
        refine_indices = set(range(len(dots_info)))
    else:
        refine_indices = {int(index) for index in refine_component_indices}
    radius_samples = [
        dots_info[index][2]
        for index in sorted(refine_indices)
        if 0 <= index < len(dots_info) and dots_info[index][2] is not None
    ]
    if not radius_samples:
        return [(cx, cy) for cx, cy, _, _ in dots_info]

    average_radius = sum(radius_samples) / len(radius_samples)
    log(f"Average geometric dot radius: {average_radius:.6f} px", verbose=True)
    refined_centers = []
    for index, (cx, cy, _, _) in enumerate(dots_info):
        refined_center = (cx, cy)
        if index not in refine_indices:
            refined_centers.append(refined_center)
            continue
        for threshold in (0.3, 0.15, 0.1, 0.05):
            next_center = second_bfs(
                int(round(refined_center[0])),
                int(round(refined_center[1])),
                refined_center[0],
                refined_center[1],
                average_radius,
                threshold,
            )
            if next_center is None:
                break
            refined_center = next_center
        refined_centers.append(refined_center)
    return refined_centers

def greedily_match_centers(true_centers, estimated_centers):
    """Greedy O(n^2) matching: repeatedly pick closest pair. Returns only matched pairs."""
    true_list = list(true_centers)
    est_list = list(estimated_centers)
    n_true = len(true_list)
    n_est = len(est_list)
    log(f"True points: {n_true}, Estimated points: {n_est}, Matched: {min(n_true, n_est)}", verbose=True)

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

def prepare_attack_run():
    if cfg.INPUT_IMAGE:
        if not os.path.isfile(cfg.INPUT_IMAGE):
            raise FileNotFoundError(f"Input map image not found: {cfg.INPUT_IMAGE}")
    else:
        utils.validate_attack_inputs()
    if not cfg.INPUT_IMAGE and (cfg.REGENERATE_BASE_MAP or not os.path.exists(cfg.FILENAME)):
        utils.render_geojson_map(cfg.JSON_FILE, cfg.FILENAME, cfg.PRIMARY_TILE_SOURCE)
    if cfg.BG_MODE:
        utils.generate_background_reference_images()
    utils.ensure_parent_dir(cfg.MANUAL_DOT_QUERIES_FILE)
    with open(cfg.MANUAL_DOT_QUERIES_FILE, "w", buffering=1) as mf:
        if cfg.CLUSTER_SIZE_MODE == "manual":
            mf.write(f"Manual dot count queries (stdin=terminal)\n{'=' * 50}\n")
        else:
            mf.write(f"Estimated cluster sizes\n{'=' * 50}\n")


def load_attack_inputs():
    image = Image.open(cfg.FILENAME).convert("RGB")
    pixels = image.load()
    width, height = image.size
    if width != cfg.WIDTH_PX or height != cfg.HEIGHT_PX:
        log(
            f"WARNING: Original image size ({width}x{height}) doesn't match expected size ({cfg.WIDTH_PX}x{cfg.HEIGHT_PX})"
        )
        log("This will cause coordinate misalignment. Please regenerate the map with option 1.")
    original_image = Image.open(cfg.FILENAME).convert("RGB")
    original_pixels = original_image.load()
    blobs = collect_dot_components(image)
    log(f"Blob pixel groups: {[len(blob) for blob in blobs]}", verbose=True)
    shape_anchor_offset = tuple(getattr(cfg, "SHAPE_OFFSET_PX", (0.0, 0.0)))
    if getattr(cfg, "CALIBRATE_SHAPE_OFFSET", False) and cfg.EVAL_SOURCE_FILE:
        shape_anchor_offset = estimate_triangle_anchor_offset(cfg.FILENAME, width, height)
    elif shape_anchor_offset != (0.0, 0.0):
        log(
            f"Applying configured shape offset: offset_x={shape_anchor_offset[0]:.3f}px, "
            f"offset_y={shape_anchor_offset[1]:.3f}px",
            verbose=False,
        )
    return {
        "image": image,
        "pixels": pixels,
        "width": width,
        "height": height,
        "original_image": original_image,
        "original_pixels": original_pixels,
        "blobs": blobs,
        "shape_anchor_offset": shape_anchor_offset,
    }


def initialize_attack_clusters(state):
    image = state["image"]
    width = state["width"]
    height = state["height"]
    original_image = state["original_image"]
    original_pixels = original_image.load()
    blobs = state["blobs"]
    shape_anchor_offset = state.get("shape_anchor_offset", (0.0, 0.0))
    pixels = image.load()
    base_background_image = None
    base_background_pixels = None
    if cfg.BG_MODE and cfg.BASE_BACKGROUND_IMG and os.path.exists(cfg.BASE_BACKGROUND_IMG):
        base_background_image = Image.open(cfg.BASE_BACKGROUND_IMG).convert("RGB")
        if base_background_image.size == original_image.size:
            base_background_pixels = base_background_image.load()
        else:
            log(
                f"Warning: background image size {base_background_image.size} does not match map size {original_image.size}; using local background estimates.",
                verbose=True,
            )

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
        fallback_centers_by_id = {}
        blob_sizes = []
        min_blob_pixels = min((len(blob) for blob in blobs), default=1)
        estimated_base_radius = estimate_blob_radius_from_pixel_count(min_blob_pixels, margin=0.5)
        circle_like_boundary_radius = estimate_blob_radius_from_pixel_count(min_blob_pixels, margin=1.5)
        circle_like_shape = is_circle_like_dot_shape()
        cluster_sizes = {}

        if cfg.CLUSTER_SIZE_MODE == "estimate":
            log(f"Minimum blob size (1-dot reference): {min_blob_pixels} dot pixels", verbose=True)
            log(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px", verbose=True)
            with open(cfg.MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                mf.write(f"Minimum blob size (1-dot reference): {min_blob_pixels} dot pixels\n")
                mf.write(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px\n")
        elif cfg.CLUSTER_SIZE_MODE == "manual":
            log(f"Minimum blob size (radius reference): {min_blob_pixels} dot pixels", verbose=True)
            log(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px", verbose=True)
            with open(cfg.MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                mf.write(f"Minimum blob size (radius reference): {min_blob_pixels} dot pixels\n")
                mf.write(f"Estimated base radius from smallest blob: {estimated_base_radius:.3f} px\n")
            if os.path.exists(cfg.CLUSTER_TYPE_GEOJSON):
                with open(cfg.CLUSTER_TYPE_GEOJSON, "r", encoding="utf-8") as cf:
                    data = json.load(cf)
                for feat in data.get("features", []):
                    props = feat.get("properties", {})
                    key = props.get("key")
                    size = props.get("size")
                    if key is not None and size is not None:
                        cluster_sizes[str(key)] = int(size)
        else:
            raise ValueError(f"Unsupported cfg.CLUSTER_SIZE_MODE: {cfg.CLUSTER_SIZE_MODE}")

        log(f"Detected {len(blobs)} dot blobs for cluster initialization.")
        for blob_index, blob in enumerate(blobs):
            log(f"Processing blob {blob_index + 1} with {len(blob)} dot pixels", verbose=True)
            if cfg.CLUSTER_SIZE_MODE == "estimate":
                cluster_ratio = len(blob) / min_blob_pixels if min_blob_pixels else 1.0
                cluster_size = estimate_cluster_count_from_blob_pixels(len(blob), min_blob_pixels)
                log(f"  Estimated {cluster_size} dots from ratio {cluster_ratio:.3f}", verbose=True)
                with open(cfg.MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                    mf.write(
                        f"Blob {blob_index + 1}: {len(blob)} dot pixels, ratio = {cluster_ratio:.3f}, estimated dots = {cluster_size}\n"
                    )
            else:
                first_pixel = min(blob)
                key = f"{first_pixel[0]},{first_pixel[1]}"
                if key in cluster_sizes:
                    cluster_size = cluster_sizes[key]
                    log(f"  (cached: {cluster_size} dots from {cfg.CLUSTER_TYPE_GEOJSON})", verbose=True)
                else:
                    img_path = save_cluster_query_preview(original_image, blob, blob_index)
                    with open(cfg.MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
                        mf.write(f"Blob {blob_index + 1}: {len(blob)} dot pixels. Image: {img_path}\n")
                        mf.flush()
                        prompt = f"Blob {blob_index + 1} ({len(blob)} dot pixels): Enter number of dots in this cluster: "
                        print(prompt, file=sys.stderr)
                        mf.write(prompt)
                        mf.flush()
                    cluster_size = max(1, int(input().strip() or "1"))
                    with open(cfg.MANUAL_DOT_QUERIES_FILE, "a", buffering=1) as mf:
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
                    utils.ensure_parent_dir(cfg.CLUSTER_TYPE_GEOJSON)
                    with open(cfg.CLUSTER_TYPE_GEOJSON, "w", encoding="utf-8") as cf:
                        json.dump({"type": "FeatureCollection", "features": features}, cf, indent=2)

            blob_sizes.append(cluster_size)
            centers = random.sample(blob, cluster_size)
            for _ in range(cfg.MAX_ITER):
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
                if all(math.hypot(a - b, c - d) < cfg.TOL for (a, c), (b, d) in zip(centers, new_centers)):
                    break
                centers = new_centers

            for center_index, (center_x, center_y) in enumerate(centers):
                assigned_dot_pixels = clusters[center_index] if center_index < len(clusters) else []
                if circle_like_shape:
                    cluster_search_radius = circle_like_boundary_radius
                else:
                    cluster_search_radius = max(
                        estimated_base_radius,
                        estimate_blob_radius_from_pixel_count(len(assigned_dot_pixels), margin=0.5),
                    )
                cx, cy = int(round(center_x)), int(round(center_y))
                queue = deque([(cx, cy)])
                seen = set()
                cluster_id = blob_index * 1000 + center_index
                cluster_size_by_id[cluster_id] = cluster_size
                fallback_centers_by_id[cluster_id] = (center_x, center_y)

                while queue:
                    x, y = queue.popleft()
                    if not (0 <= x < width and 0 <= y < height):
                        continue
                    if (x, y) in seen:
                        continue
                    if math.hypot(x - center_x, y - center_y) > cluster_search_radius:
                        continue
                    seen.add((x, y))
                    if bfs_owner[x][y] != -1 and bfs_owner[x][y] != cluster_id and not is_solid_dot_pixel(pixels[x, y]):
                        continue
                    bfs_owner[x][y] = cluster_id
                    for dx, dy in cfg.DIRS:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) not in seen and is_solid_dot_pixel(pixels[x, y]):
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

        log(f"Unassigned {len(to_unassign)} border pixels due to king-adjacency conflicts.", verbose=True)

        for x in range(width):
            for y in range(height):
                if not is_solid_dot_pixel(pixels[x, y]) and bfs_owner[x][y] != -1:
                    cluster_id = bfs_owner[x][y]
                    blob_index = cluster_id // 1000
                    center_index = cluster_id % 1000
                    pixels[x, y] = generate_cluster_fill_color(blob_index * 100 + center_index)

        if cfg.SAVE_DEBUG_ARTIFACTS:
            utils.ensure_parent_dir(cfg.BOUNDARY_PIXELS_IMG)
            utils.save_image_for_attack(image, cfg.BOUNDARY_PIXELS_IMG)
            log(f"Saved {cfg.BOUNDARY_PIXELS_IMG}", verbose=True)
        cluster_pixels = defaultdict(list)
        for x in range(width):
            for y in range(height):
                if not is_solid_dot_pixel(pixels[x, y]) and bfs_owner[x][y] != -1:
                    cluster_pixels[bfs_owner[x][y]].append((x, y))
        if getattr(cfg, "IMAGE_FORMAT", "png") == "jpeg":
            fallback_count = 0
            search_radius = max(4.0, estimated_base_radius + 2.0)
            inner_radius = max(1.0, estimated_base_radius * 0.45)
            for cid, (center_x, center_y) in fallback_centers_by_id.items():
                if len(cluster_pixels.get(cid, [])) >= 3:
                    continue
                x_min = max(0, int(math.floor(center_x - search_radius)))
                x_max = min(width - 1, int(math.ceil(center_x + search_radius)))
                y_min = max(0, int(math.floor(center_y - search_radius)))
                y_max = min(height - 1, int(math.ceil(center_y + search_radius)))
                fallback_pixels = []
                for px in range(x_min, x_max + 1):
                    for py in range(y_min, y_max + 1):
                        if is_solid_dot_pixel(pixels[px, py]):
                            continue
                        distance = math.hypot((px + 0.5) - center_x, (py + 0.5) - center_y)
                        if inner_radius <= distance <= search_radius:
                            fallback_pixels.append((px, py))
                if len(fallback_pixels) >= 3:
                    cluster_pixels[cid] = fallback_pixels
                    fallback_count += 1
            if fallback_count:
                log(f"Added JPEG boundary fallback pixels for {fallback_count} clusters.", verbose=True)
        log(f"Collected boundary pixels for {len(cluster_pixels)} clusters.", verbose=True)

        circle_centers = {}
        for cid, pixels_list in cluster_pixels.items():
            xs = [x for x, y in pixels_list]
            ys = [y for x, y in pixels_list]
            circle_centers[cid] = fit_circle(xs, ys) if len(xs) >= 3 else (None, None)

        cluster_pixel_sets = {cid: set(pixels_list) for cid, pixels_list in cluster_pixels.items()}

        def estimate_boundary_background(x, y, cid):
            if base_background_pixels is not None:
                return base_background_pixels[x, y]
            if circle_like_shape:
                return cfg.BACKGROUND_COLOR
            outside_pixels = []
            cluster_pixel_set = cluster_pixel_sets.get(cid, set())
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if (nx, ny) in cluster_pixel_set:
                        continue
                    if is_dot_pixel(original_pixels[nx, ny]):
                        continue
                    outside_pixels.append(original_pixels[nx, ny])
            return average_rgb(outside_pixels) or cfg.BACKGROUND_COLOR

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
                background_pixel = estimate_boundary_background(int(x), int(y), cid)
                denom = l1_color_distance(background_pixel, cfg.DOT_COLOR)
                if denom == 0:
                    denom = 1e-6
                ratio = l1_color_distance(px_val, background_pixel) / denom
                ratio = max(min(ratio, 1), 0.01)
                theta = math.atan2(abs(y_shifted - yc_shifted), abs(x_shifted - xc_shifted))
                signed_theta = math.atan2(y_shifted - yc_shifted, x_shifted - xc_shifted)
                try:
                    r_offset = solve_radius_for_unit_square_area(theta, ratio)
                except Exception:
                    r_offset = 0.0
                if r_offset > 5:
                    r_offset = 5
                elif r_offset < -5:
                    r_offset = -5
                radius = math.hypot(abs(x_shifted - xc_shifted), abs(y_shifted - yc_shifted)) + r_offset
                adjusted_points.append((xc_shifted + radius * math.cos(signed_theta), yc_shifted + radius * math.sin(signed_theta)))
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
    isolated_circle_like_run = (
        is_circle_like_dot_shape()
        and len(prev_centers) == len(blobs)
        and all(cluster_size_by_id.get(cid) == 1 for cid in keys)
    )
    if isolated_circle_like_run:
        geometric_background = cfg.BASE_BACKGROUND_IMG if cfg.BG_MODE else None
        geometric_centers = geometric_component_center_estimates(
            cfg.FILENAME,
            background_image_path=geometric_background,
        )
        if len(geometric_centers) == len(prev_centers):
            _, _, matched_geometric_centers, matched_prev_indices, _ = greedily_match_centers(prev_centers, geometric_centers)
            improved_centers = list(prev_centers)
            for prev_index, geometric_center in zip(matched_prev_indices, matched_geometric_centers):
                improved_centers[prev_index] = geometric_center
            prev_centers = improved_centers
            log("Using refined geometric circle centers for isolated-dot initialization.")
        else:
            log(
                f"Geometric circle initializer skipped: expected {len(prev_centers)} centers, found {len(geometric_centers)}.",
                verbose=True,
            )
    if shape_anchor_offset != (0.0, 0.0):
        prev_centers = apply_pixel_offset_to_centers(
            prev_centers,
            shape_anchor_offset[0],
            shape_anchor_offset[1],
            width=width,
            height=height,
        )
    return {
        "image": image,
        "pixels": pixels,
        "width": width,
        "height": height,
        "original_image": original_image,
        "original_pixels": original_pixels,
        "cluster_pixels": cluster_pixels,
        "prev_centers": prev_centers,
        "keys": keys,
    }


def format_trace_value(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8f}"


def average_or_none(values):
    return sum(values) / len(values) if values else None


def write_descent_trace_row(
    trace_file,
    phase,
    step_index,
    step_size,
    centers,
    width,
    height,
    direction_counts=None,
    chosen_errors=None,
    same_errors=None,
):
    if trace_file is None:
        return
    direction_counts = direction_counts or {}
    chosen_errors = chosen_errors or []
    same_errors = same_errors or []
    objective_gains = [same - chosen for same, chosen in zip(same_errors, chosen_errors)]
    values = [
        phase,
        step_index,
        step_size,
        len(centers),
        sum(count for direction, count in direction_counts.items() if direction != "same"),
        direction_counts.get("left", 0),
        direction_counts.get("right", 0),
        direction_counts.get("up", 0),
        direction_counts.get("down", 0),
        direction_counts.get("up_left", 0),
        direction_counts.get("up_right", 0),
        direction_counts.get("down_left", 0),
        direction_counts.get("down_right", 0),
        direction_counts.get("same", 0),
        average_or_none(chosen_errors),
        average_or_none(same_errors),
        average_or_none(objective_gains),
    ]
    trace_file.write(",".join(format_trace_value(value) if not isinstance(value, str) else value for value in values) + "\n")


def run_perceptual_descent(state):
    width = state["width"]
    height = state["height"]
    original_pixels = state["original_pixels"]
    cluster_pixels = state["cluster_pixels"]
    prev_centers = state["prev_centers"]
    keys = state["keys"]

    base_background_pixels = None
    shifted_background_pixels = None
    if cfg.BG_MODE:
        base_background_pixels = Image.open(cfg.BASE_BACKGROUND_IMG).convert("RGB").load()
        shifted_background_pixels = Image.open(cfg.SHIFTED_BACKGROUND_IMG).convert("RGB").load()

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
                        if 0 <= kx < width and 0 <= ky < height and is_solid_dot_pixel(new[kx, ky]):
                            has_dot_neighbor = True
                            break
                    if not has_dot_neighbor:
                        neighbor_pixels.append((nx, ny))
                if cfg.BG_MODE:
                    new_background = shifted_background_pixels[x, y]
                    base_background = base_background_pixels[x, y]
                else:
                    new_background = cfg.BACKGROUND_COLOR
                    base_background = cfg.BACKGROUND_COLOR
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
                d_new_bg_dot = l1_color_distance(new_background, cfg.DOT_COLOR)
                d_base_bg_dot = l1_color_distance(base_background, cfg.DOT_COLOR)
                if d_new_bg_dot == 0 or d_base_bg_dot == 0:
                    continue
                new_term = l1_color_distance(new[x, y], new_background) / d_new_bg_dot
                base_term = l1_color_distance(base[x, y], base_background) / d_base_bg_dot
                error += abs(new_term - base_term)
            scores[i] = error
        return scores

    trace_header = [
        "phase",
        "step",
        "step_size_px",
        "center_count",
        "moved_count",
        "left_count",
        "right_count",
        "up_count",
        "down_count",
        "up_left_count",
        "up_right_count",
        "down_left_count",
        "down_right_count",
        "same_count",
        "avg_chosen_objective",
        "avg_same_objective",
        "avg_objective_gain",
    ]
    trace_file = None
    if cfg.SAVE_DEBUG_ARTIFACTS:
        utils.ensure_parent_dir(cfg.DESCENT_TRACE_FILE)
        trace_file = open(cfg.DESCENT_TRACE_FILE, "w", encoding="utf-8", buffering=1)
        trace_file.write(",".join(trace_header) + "\n")
    try:
        step_size = cfg.INITIAL_STEP_SIZE
        step_index = 0
        write_descent_trace_row(trace_file, "initial", step_index, step_size, prev_centers, width, height)
        while step_size > cfg.MIN_STEP_SIZE:
            step_index += 1
            candidate_specs = [
                ("left", -step_size, 0.0, cfg.LEFT_GEOJSON, cfg.LEFT_IMG),
                ("right", step_size, 0.0, cfg.RIGHT_GEOJSON, cfg.RIGHT_IMG),
                ("up", 0.0, step_size, cfg.UP_GEOJSON, cfg.UP_IMG),
                ("down", 0.0, -step_size, cfg.DOWN_GEOJSON, cfg.DOWN_IMG),
            ]
            if getattr(cfg, "IMAGE_FORMAT", "png") == "jpeg":
                candidate_specs.extend(
                    [
                        ("up_left", -step_size, step_size, cfg.UP_LEFT_GEOJSON, cfg.UP_LEFT_IMG),
                        ("up_right", step_size, step_size, cfg.UP_RIGHT_GEOJSON, cfg.UP_RIGHT_IMG),
                        ("down_left", -step_size, -step_size, cfg.DOWN_LEFT_GEOJSON, cfg.DOWN_LEFT_IMG),
                        ("down_right", step_size, -step_size, cfg.DOWN_RIGHT_GEOJSON, cfg.DOWN_RIGHT_IMG),
                    ]
                )

            candidate_centers = {}
            candidate_errors = {}
            for direction_name, dx, dy, geojson_path, image_path in candidate_specs:
                centers = [(prev_centers[i][0] + dx, prev_centers[i][1] + dy) for i in range(len(keys))]
                coords = [utils.pixel_to_geographic(c[0], c[1], width, height) for c in centers]
                utils.write_point_geojson(coords, geojson_path)
                utils.render_geojson_map(geojson_path, image_path, cfg.SHIFTED_TILE_SOURCE)
                candidate_centers[direction_name] = centers
                candidate_errors[direction_name] = calc_error_for_dot(
                    original_pixels,
                    utils.load_rgb_image_retry(image_path).load(),
                )

            no_change = [(prev_centers[i][0], prev_centers[i][1]) for i in range(len(keys))]
            no_change_coords = [utils.pixel_to_geographic(c[0], c[1], width, height) for c in no_change]
            utils.write_point_geojson(no_change_coords, cfg.NOCHANGE_GEOJSON)
            utils.render_geojson_map(cfg.NOCHANGE_GEOJSON, cfg.NOCHANGE_IMG, cfg.SHIFTED_TILE_SOURCE)
            no_change_err = calc_error_for_dot(original_pixels, utils.load_rgb_image_retry(cfg.NOCHANGE_IMG).load())

            updated_centers = [(0, 0) for _ in range(len(keys))]
            direction_counts = defaultdict(int)
            chosen_errors = []
            same_errors = []
            for i in range(len(keys)):
                ordered_choices = [
                    (candidate_direction, candidate_errors[candidate_direction][i], candidate_centers[candidate_direction][i])
                    for candidate_direction, _, _, _, _ in candidate_specs
                ]
                ordered_choices.append(("same", no_change_err[i], no_change[i]))
                direction, min_err, chosen_center = min(ordered_choices, key=lambda item: item[1])
                direction_counts[direction] += 1
                chosen_errors.append(min_err)
                same_errors.append(no_change_err[i])
                updated_centers[i] = chosen_center
            prev_centers = updated_centers
            write_descent_trace_row(
                trace_file,
                "iteration",
                step_index,
                step_size,
                prev_centers,
                width,
                height,
                direction_counts=direction_counts,
                chosen_errors=chosen_errors,
                same_errors=same_errors,
            )
            step_size /= cfg.STEP_DIVISOR
            log(f"Next step size: {step_size:.8f}", verbose=True)
        write_descent_trace_row(trace_file, "final", step_index, step_size, prev_centers, width, height)
    finally:
        if trace_file is not None:
            trace_file.close()

    state.update(
        {
            "prev_centers": prev_centers,
            "cluster_pixels": cluster_pixels,
        }
    )
    return state


def write_attack_results(state):
    width = state["width"]
    height = state["height"]
    recovered = [
        utils.pixel_to_geographic(x, y, width, height)
        for x, y in state["prev_centers"]
    ]
    utils.write_point_geojson(recovered, cfg.RECOVERED_GEOJSON)
    log(f"Recovered {len(recovered)} locations -> {cfg.RECOVERED_GEOJSON}")
    return cfg.RECOVERED_GEOJSON


def main():
    random.seed(cfg.RANDOM_SEED)
    prepare_attack_run()
    state = load_attack_inputs()
    state = initialize_attack_clusters(state)
    state = run_perceptual_descent(state)
    return write_attack_results(state)


def run_cli(argv=None):
    configure_cli(argv)
    for dataset_key in cfg.RUN_DATASET:
        cfg.configure_dataset(dataset_key)
        try:
            log(f"Attacking {cfg.FILENAME}")
            main()
        finally:
            if cfg.TEMP_WORK_ROOT:
                shutil.rmtree(cfg.TEMP_WORK_ROOT, ignore_errors=True)


if __name__ == "__main__":
    run_cli()

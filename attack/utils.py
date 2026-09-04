import json
import os
import tempfile
import time
from difflib import get_close_matches

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
from PIL import Image

from . import config as cfg


def _is_verbose():
    return getattr(cfg, "LOG_LEVEL", "summary") == "verbose"


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(os.fspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def is_dot_pixel(pixel, dot_color=None):
    rgb = tuple(int(value) for value in pixel[:3])
    dot_rgb = tuple(dot_color or cfg.DOT_COLOR)
    if rgb == dot_rgb:
        return True

    if getattr(cfg, "IMAGE_FORMAT", "png") != "jpeg":
        return False

    background_rgb = tuple(getattr(cfg, "BACKGROUND_COLOR", (255, 255, 255)))
    dot_distance = sum(abs(channel - target) for channel, target in zip(rgb, dot_rgb))
    background_distance = sum(abs(channel - target) for channel, target in zip(rgb, background_rgb))
    dot_background_distance = sum(abs(dot - background) for dot, background in zip(dot_rgb, background_rgb))
    return (
        dot_background_distance > 0
        and dot_distance <= max(80, 0.35 * dot_background_distance)
        and background_distance >= max(40, 0.25 * dot_background_distance)
        and dot_distance < background_distance
    )


def is_solid_dot_pixel(pixel, dot_color=None):
    return is_dot_pixel(pixel, dot_color=dot_color)


def save_image_for_attack(image, output_path):
    ensure_parent_dir(output_path)
    if getattr(cfg, "IMAGE_FORMAT", "png") == "jpeg":
        if image.mode in ("RGBA", "LA") or ("transparency" in image.info):
            flattened = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.convert("RGBA").getchannel("A")
            flattened.paste(image.convert("RGB"), mask=alpha)
        else:
            flattened = image.convert("RGB")
        flattened.save(
            output_path,
            format="JPEG",
            quality=int(getattr(cfg, "JPEG_QUALITY", 80)),
            subsampling=0,
        )
    else:
        image.save(output_path)


def resolve_geojson_input_path(path: str) -> str:
    if os.path.exists(path):
        return path

    geojson_candidates = []
    for root, _, files in os.walk(cfg.BASE_DIR):
        for name in files:
            if name.lower().endswith(".geojson"):
                geojson_candidates.append(os.path.relpath(os.path.join(root, name), cfg.BASE_DIR))
    suggestions = get_close_matches(path, geojson_candidates, n=5, cutoff=0.5)
    suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise FileNotFoundError(f"GeoJSON file not found: {path}.{suggestion_text}")


def resolve_dataset_input_path(path: str) -> str:
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(os.path.join(cfg.DATA_ROOT, path))
        candidates.append(os.path.join(cfg.BASE_DIR, path))

    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if os.path.exists(normalized):
            return normalized

    checked_paths = ", ".join(os.path.normpath(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Required input file not found: {path}. Checked: {checked_paths}")


def validate_attack_inputs():
    if not cfg.JSON_FILE or not os.path.exists(cfg.JSON_FILE):
        raise FileNotFoundError(f"Source GeoJSON not found: {cfg.JSON_FILE}")


def write_point_geojson(latlon_list, output_path):
    ensure_parent_dir(output_path)
    geojson_data = {"type": "FeatureCollection", "features": []}
    for lat, lon in latlon_list:
        geojson_data["features"].append(
            {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [lon, lat]}}
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)


def load_ground_truth_points(filename=None):
    """Load point coordinates for legacy calibration/evaluation callers only."""
    filename = filename or getattr(cfg, "EVAL_SOURCE_FILE", None)
    if not filename:
        raise ValueError("No evaluation source was configured.")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        (coords[0], coords[1])
        for feature in data.get("features", [])
        for coords in [feature.get("geometry", {}).get("coordinates")]
        if coords and len(coords) >= 2
    ]


def geographic_to_pixel(lat, lon, img_width, img_height):
    x = (lon - cfg.MIN_LON) / (cfg.MAX_LON - cfg.MIN_LON) * img_width
    y = (cfg.MAX_LAT - lat) / (cfg.MAX_LAT - cfg.MIN_LAT) * img_height
    return (x, y)


def pixel_to_geographic(x, y, img_width, img_height):
    lon = cfg.MIN_LON + (x / img_width) * (cfg.MAX_LON - cfg.MIN_LON)
    lat = cfg.MAX_LAT - (y / img_height) * (cfg.MAX_LAT - cfg.MIN_LAT)
    return (lat, lon)


def load_rgb_image_retry(image_path, retries=5, delay=0.2):
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


def render_geojson_map_with_geopandas(predicted_points: str, output_path: str, tile_source=None):
    os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()

    predicted_points = resolve_geojson_input_path(predicted_points)
    gdf = gpd.read_file(predicted_points)
    gdf = gdf[
        gdf.geometry.notnull()
        & gdf.geometry.x.notnull()
        & gdf.geometry.y.notnull()
        & gdf.geometry.y.between(-89.999, 89.999)
    ]

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326)

    tile_source = cfg.PRIMARY_TILE_SOURCE if tile_source is None else tile_source
    use_blank_white_background = tile_source == "white"
    ctx.set_cache_dir(os.path.join(cfg.BASE_DIR, "osm_cache"))

    stretch_factor = 1
    fig_w_in = (cfg.WIDTH_PX / 96) * stretch_factor
    fig_h_in = cfg.HEIGHT_PX / 96
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=96)
    if use_blank_white_background:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

    mm_to_pt = 72 / 25.4
    marker_diameter_mm = cfg.DOT_RADIUS_MM * 2
    marker_size_pts2 = (marker_diameter_mm * mm_to_pt) ** 2
    marker_shape = cfg.DOT_SHAPE if cfg.DOT_SHAPE is not None else "o"

    if not gdf.empty:
        gdf.plot(
            ax=ax,
            color="#ff0000",
            markersize=marker_size_pts2,
            marker=marker_shape,
            edgecolor="none",
            linewidth=0,
            alpha=0 if "dummy_points.geojson" in predicted_points else 1,
        )
    elif _is_verbose():
        print(f"Warning: no valid points found in {predicted_points}; rendering basemap only.", flush=True)

    ax.set_xlim(cfg.MIN_LON, cfg.MAX_LON)
    ax.set_ylim(cfg.MIN_LAT, cfg.MAX_LAT)
    if not use_blank_white_background:
        ctx.add_basemap(ax, source=tile_source, crs="EPSG:4326", zoom=cfg.MAP_ZOOM, reset_extent=False)
    ax.set_xlim(cfg.MIN_LON, cfg.MAX_LON)
    ax.set_ylim(cfg.MIN_LAT, cfg.MAX_LAT)
    ax.set_aspect(1 / stretch_factor)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ensure_parent_dir(output_path)
    save_facecolor = "white" if use_blank_white_background else "none"
    save_kwargs = {"dpi": 96, "bbox_inches": None, "pad_inches": 0, "facecolor": save_facecolor, "format": "PNG"}
    if getattr(cfg, "IMAGE_FORMAT", "png") == "jpeg":
        parent_dir = os.path.dirname(os.fspath(output_path)) or "."
        tmp_png = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", dir=parent_dir, delete=False) as tmp_file:
                tmp_png = tmp_file.name
            fig.savefig(tmp_png, **save_kwargs)
            with Image.open(tmp_png) as rendered:
                save_image_for_attack(rendered.convert("RGB"), output_path)
        finally:
            if tmp_png and os.path.exists(tmp_png):
                os.remove(tmp_png)
    else:
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)
    if _is_verbose():
        print(f"Map saved to: {output_path}")


def render_geojson_map(predicted_points: str, output_path: str, tile_source=None):
    return render_geojson_map_with_geopandas(predicted_points, output_path, tile_source=tile_source)


def generate_background_reference_images():
    write_point_geojson([], cfg.BACKGROUND_REFERENCE_GEOJSON)
    render_geojson_map(cfg.BACKGROUND_REFERENCE_GEOJSON, cfg.BASE_BACKGROUND_IMG, cfg.PRIMARY_TILE_SOURCE)
    render_geojson_map(cfg.BACKGROUND_REFERENCE_GEOJSON, cfg.SHIFTED_BACKGROUND_IMG, cfg.SHIFTED_TILE_SOURCE)

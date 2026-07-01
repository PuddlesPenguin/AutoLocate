import argparse
import os

from pathlib import Path


DEFAULT_RUN_DATASET = {
    "OpenAddresses": "OpenAddress/US.geojson",
    "Synthetic": "Synthetic/US.geojson",
}

DEFAULT_EVAL_SOURCE_FILES = {
    "OpenAddresses": "OpenAddress/US.geojson",
    "Synthetic": "Synthetic/US.geojson",
}

DEFAULT_TEST_NAME = "192dpiSatelliteBG-US"
DEFAULT_EVAL_DECIMALS = 6
DEFAULT_CLUSTER_TYPE = "new"
DEFAULT_CLUSTER_SIZE_MODE = "estimate"
DEFAULT_BG_MODE = False
DEFAULT_REGENERATE_BASE_MAP = True
DEFAULT_WIDTH_PX = 2284
DEFAULT_HEIGHT_PX = 1424
DEFAULT_DOT_RADIUS_MM = 2
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-2
DEFAULT_INITIAL_STEP_SIZE = 0.5
DEFAULT_STEP_DIVISOR = 1.2
DEFAULT_MIN_STEP_SIZE = 0.0001
DEFAULT_PIXEL_SIZE = 0.02587884152408056
DEFAULT_MIN_LON = -126.17658145147592563
DEFAULT_MAX_LAT = 58.62037301762128294
DEFAULT_MAP_ZOOM = 5
DEFAULT_DOT_SHAPE = None
DEFAULT_LOG_LEVEL = "summary"
DEFAULT_SHAPE_OFFSET_PX = (0.0, 0.0)
DEFAULT_CALIBRATE_SHAPE_OFFSET = False
DEFAULT_USE_GEOMETRIC_CIRCLE_INIT = False
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_JPEG_QUALITY = 80

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "CoordinateJSONs"

RUN_DATASET = dict(DEFAULT_RUN_DATASET)
EVAL_SOURCE_FILES = dict(DEFAULT_EVAL_SOURCE_FILES)
TEST_NAME = DEFAULT_TEST_NAME
EVAL_DECIMALS = DEFAULT_EVAL_DECIMALS
CLUSTER_TYPE = DEFAULT_CLUSTER_TYPE
CLUSTER_SIZE_MODE = DEFAULT_CLUSTER_SIZE_MODE
BG_MODE = DEFAULT_BG_MODE
REGENERATE_BASE_MAP = DEFAULT_REGENERATE_BASE_MAP
WIDTH_PX = DEFAULT_WIDTH_PX
HEIGHT_PX = DEFAULT_HEIGHT_PX
DOT_RADIUS_MM = DEFAULT_DOT_RADIUS_MM
MAX_ITER = DEFAULT_MAX_ITER
TOL = DEFAULT_TOL
INITIAL_STEP_SIZE = DEFAULT_INITIAL_STEP_SIZE
STEP_DIVISOR = DEFAULT_STEP_DIVISOR
MIN_STEP_SIZE = DEFAULT_MIN_STEP_SIZE
PIXEL_SIZE = DEFAULT_PIXEL_SIZE
MIN_LON = DEFAULT_MIN_LON
MAX_LAT = DEFAULT_MAX_LAT
MAP_ZOOM = DEFAULT_MAP_ZOOM
MAX_LON = MIN_LON + PIXEL_SIZE * WIDTH_PX
MIN_LAT = MAX_LAT - PIXEL_SIZE * HEIGHT_PX
DOT_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)
PRIMARY_TILE_SOURCE = None
SHIFTED_TILE_SOURCE = None
DOT_SHAPE = DEFAULT_DOT_SHAPE
LOG_LEVEL = DEFAULT_LOG_LEVEL
SHAPE_OFFSET_PX = DEFAULT_SHAPE_OFFSET_PX
CALIBRATE_SHAPE_OFFSET = DEFAULT_CALIBRATE_SHAPE_OFFSET
USE_GEOMETRIC_CIRCLE_INIT = DEFAULT_USE_GEOMETRIC_CIRCLE_INIT
IMAGE_FORMAT = DEFAULT_IMAGE_FORMAT
JPEG_QUALITY = DEFAULT_JPEG_QUALITY
DIRS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

CURRENT_DATASET_KEY = None
JSON_FILE = None
EVAL_SOURCE_FILE = None
RESULTS_ROOT = None
AUGMENTED_ROOT = None
RUN_SUFFIX = None
RESULTS_RUN_DIR = None
AUGMENTED_RUN_DIR = None
FILENAME = None
EVAL_JSON = None
MANUAL_DOT_QUERIES_FILE = None
CLUSTER_QUERY_IMAGE_PREFIX = None
CLUSTER_QUERY_IMAGE_PATH = None
BACKGROUND_REFERENCE_GEOJSON = None
BASE_BACKGROUND_IMG = None
SHIFTED_BACKGROUND_IMG = None
CLUSTER_TYPE_ROOT = None
CLUSTER_TYPE_GEOJSON = None
LEFT_GEOJSON = None
RIGHT_GEOJSON = None
UP_GEOJSON = None
DOWN_GEOJSON = None
UP_LEFT_GEOJSON = None
UP_RIGHT_GEOJSON = None
DOWN_LEFT_GEOJSON = None
DOWN_RIGHT_GEOJSON = None
NOCHANGE_GEOJSON = None
LEFT_IMG = None
RIGHT_IMG = None
UP_IMG = None
DOWN_IMG = None
UP_LEFT_IMG = None
UP_RIGHT_IMG = None
DOWN_LEFT_IMG = None
DOWN_RIGHT_IMG = None
NOCHANGE_IMG = None
BOUNDARY_PIXELS_IMG = None
GEO_PLOT_FILE = None
GEO_BOX_PDF_FILE = None
DOT_RESULTS_FILE = None
SUMMARY_RESULTS_FILE = None
RUN_LOG_FILE = None
RESULTS_RUN_LOG_FILE = None
DESCENT_TRACE_FILE = None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the AutoLocate attack pipeline with configurable runtime settings."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=tuple(DEFAULT_RUN_DATASET.keys()),
        help="Optional subset of datasets to run. Omit to run all configured datasets.",
    )
    parser.add_argument("--test-name", default=DEFAULT_TEST_NAME)
    parser.add_argument("--eval-decimals", type=int, default=DEFAULT_EVAL_DECIMALS)
    parser.add_argument("--cluster-type", default=DEFAULT_CLUSTER_TYPE)
    parser.add_argument(
        "--cluster-size-mode",
        choices=("manual", "estimate"),
        default=DEFAULT_CLUSTER_SIZE_MODE,
    )
    parser.add_argument("--bg-mode", action="store_true", default=DEFAULT_BG_MODE)
    parser.add_argument(
        "--regenerate-base-map",
        dest="regenerate_base_map",
        action="store_true",
        default=DEFAULT_REGENERATE_BASE_MAP,
    )
    parser.add_argument(
        "--no-regenerate-base-map",
        dest="regenerate_base_map",
        action="store_false",
    )
    parser.add_argument("--width-px", type=int, default=DEFAULT_WIDTH_PX)
    parser.add_argument("--height-px", type=int, default=DEFAULT_HEIGHT_PX)
    parser.add_argument("--dot-radius-mm", type=float, default=DEFAULT_DOT_RADIUS_MM)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--initial-step-size", type=float, default=DEFAULT_INITIAL_STEP_SIZE)
    parser.add_argument("--step-divisor", type=float, default=DEFAULT_STEP_DIVISOR)
    parser.add_argument("--min-step-size", type=float, default=DEFAULT_MIN_STEP_SIZE)
    parser.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE)
    parser.add_argument("--min-lon", type=float, default=DEFAULT_MIN_LON)
    parser.add_argument("--max-lat", type=float, default=DEFAULT_MAX_LAT)
    parser.add_argument("--map-zoom", type=int, default=DEFAULT_MAP_ZOOM)
    parser.add_argument(
        "--dot-shape",
        default="circle",
        help='Dot marker shape. Use "circle", "triangle", "pentagon", or a tuple like "3,0,0".',
    )
    parser.add_argument(
        "--log-level",
        choices=("summary", "verbose"),
        default=DEFAULT_LOG_LEVEL,
        help="Controls how much progress detail is printed during a run.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpeg"),
        default=DEFAULT_IMAGE_FORMAT,
        help="Rendered attack image format. PNG is the default; JPEG enables compressed-dot detection.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality used when --image-format jpeg.",
    )
    parser.add_argument(
        "--shape-offset-px",
        nargs=2,
        type=float,
        metavar=("DX", "DY"),
        default=DEFAULT_SHAPE_OFFSET_PX,
        help="Fixed pixel offset applied to detected centers before descent, mainly for marker-anchor experiments.",
    )
    parser.add_argument(
        "--calibrate-shape-offset",
        action="store_true",
        default=DEFAULT_CALIBRATE_SHAPE_OFFSET,
        help="Estimate a triangle marker offset from evaluation truth for diagnostics.",
    )
    parser.add_argument(
        "--use-geometric-circle-init",
        action="store_true",
        default=DEFAULT_USE_GEOMETRIC_CIRCLE_INIT,
        help="Initialize isolated circle-dot runs from the refined geometric estimator instead of the modified-method boundary initializer.",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def parse_dot_shape(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "circle", "o", "round"}:
        return None
    if normalized == "triangle":
        return (3, 0, 0)
    if normalized == "pentagon":
        return (5, 0, 0)
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) == 3:
        try:
            return tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid dot shape tuple: {value}") from exc
    raise ValueError(f"Unsupported dot shape value: {value}")


def select_datasets(mapping, selected_names):
    if not selected_names:
        return dict(mapping)

    filtered = {name: mapping[name] for name in selected_names if name in mapping}
    if not filtered:
        raise ValueError("No valid datasets were selected.")
    return filtered


def apply_args(args):
    global RUN_DATASET, EVAL_SOURCE_FILES, TEST_NAME, EVAL_DECIMALS, CLUSTER_TYPE, CLUSTER_SIZE_MODE
    global BG_MODE, REGENERATE_BASE_MAP, WIDTH_PX, HEIGHT_PX, DOT_RADIUS_MM, MAX_ITER, TOL
    global INITIAL_STEP_SIZE, STEP_DIVISOR, MIN_STEP_SIZE, PIXEL_SIZE, MIN_LON, MAX_LAT, MAP_ZOOM
    global MAX_LON, MIN_LAT, DOT_SHAPE, LOG_LEVEL
    global SHAPE_OFFSET_PX, CALIBRATE_SHAPE_OFFSET, USE_GEOMETRIC_CIRCLE_INIT, IMAGE_FORMAT, JPEG_QUALITY

    RUN_DATASET = select_datasets(DEFAULT_RUN_DATASET, args.datasets)
    EVAL_SOURCE_FILES = select_datasets(DEFAULT_EVAL_SOURCE_FILES, args.datasets)
    TEST_NAME = args.test_name
    EVAL_DECIMALS = args.eval_decimals
    CLUSTER_TYPE = args.cluster_type
    CLUSTER_SIZE_MODE = args.cluster_size_mode
    BG_MODE = args.bg_mode
    REGENERATE_BASE_MAP = args.regenerate_base_map
    WIDTH_PX = args.width_px
    HEIGHT_PX = args.height_px
    DOT_RADIUS_MM = args.dot_radius_mm
    MAX_ITER = args.max_iter
    TOL = args.tol
    INITIAL_STEP_SIZE = args.initial_step_size
    STEP_DIVISOR = args.step_divisor
    MIN_STEP_SIZE = args.min_step_size
    PIXEL_SIZE = args.pixel_size
    MIN_LON = args.min_lon
    MAX_LAT = args.max_lat
    MAP_ZOOM = args.map_zoom
    DOT_SHAPE = parse_dot_shape(args.dot_shape)
    LOG_LEVEL = args.log_level
    IMAGE_FORMAT = args.image_format
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")
    JPEG_QUALITY = args.jpeg_quality
    SHAPE_OFFSET_PX = tuple(args.shape_offset_px)
    CALIBRATE_SHAPE_OFFSET = args.calibrate_shape_offset
    USE_GEOMETRIC_CIRCLE_INIT = args.use_geometric_circle_init
    MAX_LON = MIN_LON + PIXEL_SIZE * WIDTH_PX
    MIN_LAT = MAX_LAT - PIXEL_SIZE * HEIGHT_PX


def configure_dataset(dataset_key):
    global CURRENT_DATASET_KEY, JSON_FILE, EVAL_SOURCE_FILE, RESULTS_ROOT, AUGMENTED_ROOT, RUN_SUFFIX
    global RESULTS_RUN_DIR, AUGMENTED_RUN_DIR, FILENAME, EVAL_JSON, MANUAL_DOT_QUERIES_FILE
    global CLUSTER_QUERY_IMAGE_PREFIX, CLUSTER_QUERY_IMAGE_PATH, BACKGROUND_REFERENCE_GEOJSON
    global BASE_BACKGROUND_IMG, SHIFTED_BACKGROUND_IMG, CLUSTER_TYPE_ROOT, CLUSTER_TYPE_GEOJSON
    global LEFT_GEOJSON, RIGHT_GEOJSON, UP_GEOJSON, DOWN_GEOJSON, NOCHANGE_GEOJSON
    global UP_LEFT_GEOJSON, UP_RIGHT_GEOJSON, DOWN_LEFT_GEOJSON, DOWN_RIGHT_GEOJSON
    global LEFT_IMG, RIGHT_IMG, UP_IMG, DOWN_IMG, NOCHANGE_IMG, BOUNDARY_PIXELS_IMG
    global UP_LEFT_IMG, UP_RIGHT_IMG, DOWN_LEFT_IMG, DOWN_RIGHT_IMG
    global GEO_PLOT_FILE, GEO_BOX_PDF_FILE, DOT_RESULTS_FILE, SUMMARY_RESULTS_FILE
    global RUN_LOG_FILE, RESULTS_RUN_LOG_FILE, DESCENT_TRACE_FILE

    CURRENT_DATASET_KEY = dataset_key
    RUN_SUFFIX = f"_{TEST_NAME}" if TEST_NAME else ""
    RESULTS_ROOT = os.path.join(BASE_DIR, "Results")
    AUGMENTED_ROOT = os.path.join(BASE_DIR, "AugmentedFiles")
    CLUSTER_TYPE_ROOT = os.path.join(AUGMENTED_ROOT, "cluster_types")
    os.makedirs(CLUSTER_TYPE_ROOT, exist_ok=True)

    def resolve(path):
        if os.path.isabs(path):
            return path
        candidates = [os.path.join(DATA_ROOT, path), os.path.join(BASE_DIR, path)]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    JSON_FILE = resolve(RUN_DATASET[dataset_key])
    EVAL_SOURCE_FILE = resolve(EVAL_SOURCE_FILES.get(dataset_key, RUN_DATASET[dataset_key]))
    RESULTS_RUN_DIR = os.path.join(RESULTS_ROOT, TEST_NAME or "default", dataset_key)
    AUGMENTED_RUN_DIR = os.path.join(AUGMENTED_ROOT, TEST_NAME or "default", dataset_key)
    os.makedirs(RESULTS_RUN_DIR, exist_ok=True)
    os.makedirs(AUGMENTED_RUN_DIR, exist_ok=True)

    dataset_suffix = f"{RUN_SUFFIX}_{dataset_key}" if RUN_SUFFIX else f"_{dataset_key}"
    image_extension = "jpeg" if IMAGE_FORMAT == "jpeg" else "png"
    FILENAME = os.path.join(RESULTS_RUN_DIR, f"map{dataset_suffix}.{image_extension}")
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
    UP_LEFT_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_up_left{dataset_suffix}.geojson")
    UP_RIGHT_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_up_right{dataset_suffix}.geojson")
    DOWN_LEFT_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_down_left{dataset_suffix}.geojson")
    DOWN_RIGHT_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_down_right{dataset_suffix}.geojson")
    NOCHANGE_GEOJSON = os.path.join(AUGMENTED_RUN_DIR, f"_no_change{dataset_suffix}.geojson")
    LEFT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"left_img{dataset_suffix}.{image_extension}")
    RIGHT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"right_img{dataset_suffix}.{image_extension}")
    UP_IMG = os.path.join(AUGMENTED_RUN_DIR, f"up_img{dataset_suffix}.{image_extension}")
    DOWN_IMG = os.path.join(AUGMENTED_RUN_DIR, f"down_img{dataset_suffix}.{image_extension}")
    UP_LEFT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"up_left_img{dataset_suffix}.{image_extension}")
    UP_RIGHT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"up_right_img{dataset_suffix}.{image_extension}")
    DOWN_LEFT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"down_left_img{dataset_suffix}.{image_extension}")
    DOWN_RIGHT_IMG = os.path.join(AUGMENTED_RUN_DIR, f"down_right_img{dataset_suffix}.{image_extension}")
    NOCHANGE_IMG = os.path.join(AUGMENTED_RUN_DIR, f"no_change_img{dataset_suffix}.{image_extension}")
    BOUNDARY_PIXELS_IMG = os.path.join(AUGMENTED_RUN_DIR, f"BoundaryPixels{dataset_suffix}.{image_extension}")
    GEO_PLOT_FILE = os.path.join(RESULTS_RUN_DIR, f"geo_error_histogram_boxplot{dataset_suffix}.{image_extension}")
    GEO_BOX_PDF_FILE = os.path.join(RESULTS_RUN_DIR, f"geo_error_boxplot{dataset_suffix}.pdf")
    DOT_RESULTS_FILE = os.path.join(RESULTS_RUN_DIR, f"dot_center_results{dataset_suffix}.txt")
    SUMMARY_RESULTS_FILE = os.path.join(RESULTS_RUN_DIR, f"summary_results{dataset_suffix}.txt")
    RUN_LOG_FILE = os.path.join(AUGMENTED_RUN_DIR, f"run_log{dataset_suffix}.txt")
    RESULTS_RUN_LOG_FILE = os.path.join(RESULTS_RUN_DIR, f"run_log{dataset_suffix}.txt")
    DESCENT_TRACE_FILE = os.path.join(RESULTS_RUN_DIR, f"descent_trace{dataset_suffix}.csv")
    CLUSTER_TYPE_GEOJSON = os.path.join(CLUSTER_TYPE_ROOT, f"{CLUSTER_TYPE}.geojson")

# AutoLocate

This repository contains research code for recovering high-precision point locations from dot maps. The current main entry point is `Attack.py`, which renders or reuses a configured map, detects red dot clusters, refines their centers with perceptual descent, and converts recovered pixel centers back to geographic coordinates.

**Quick Summary**:
- **Purpose**: Recover geographic point locations from rendered dot maps by estimating dot centers at subpixel precision and mapping them back to latitude/longitude.
- **Main script**: `Attack.py`
- **Final outputs**: `Results/<TEST_NAME>/<dataset>/`
- **Intermediate artifacts**: `AugmentedFiles/<TEST_NAME>/<dataset>/`

**Requirements**:
- **Python**: 3.8+ recommended.
- **Current attack/defense dependencies**: install from `requirements.txt`
  - `Pillow`, `scipy`, `geopy`, `numpy`, `pyproj`, `geopandas`, `matplotlib`, `contextily`, `xyzservices`, `rasterio`
- Older helper/baseline scripts may need extra geospatial packages depending on your environment.

Note: `Attack.py` no longer uses the old `1` / `2` menu flow. A single run now ensures the base map exists (or regenerates it when `REGENERATE_BASE_MAP = True`) and then runs the attack.

**What each script does**
- `Attack.py`: Current cleaned attack pipeline for connected dot clusters.
  - Inputs: edit the top-level constants in the file, especially `RUN_DATASET`, `EVAL_SOURCE_FILES`, `TEST_NAME`, `CLUSTER_TYPE`, `CLUSTER_SIZE_MODE`, `BG_MODE`, `REGENERATE_BASE_MAP`, rendering settings, optimization settings, and map bounds.
  - Outputs: final artifacts in `Results/<TEST_NAME>/<dataset>/`, including the rendered map, rounded evaluation GeoJSON, per-dot results, summary metrics, and geodesic error plots.
  - Intermediate artifacts: candidate GeoJSONs and rendered candidate images, `BoundaryPixels*.png`, background-reference images, `manual_dot_queries*.txt`, `cluster_size*.png`, and `run_log*.txt` in `AugmentedFiles/<TEST_NAME>/<dataset>/`.
  - Notes: `CLUSTER_SIZE_MODE = "estimate"` estimates dots per red blob automatically. `CLUSTER_SIZE_MODE = "manual"` prompts for cluster sizes and caches them in `AugmentedFiles/cluster_types/<CLUSTER_TYPE>.geojson`.

- `attack_parser.py`: CLI parser, defaults, and runtime dataset/path configuration. Key flags include `--datasets`, `--test-name`, `--dot-shape`, and `--dot-radius-mm`.

- `attack_utils.py`: Utility helpers for GeoJSON, GeoPandas map rendering, PNG/JPEG output, validation, and result writing.
- `Defense.py`: Optional defense/quantization tool. Pass a GeoTIFF explicitly with `--raster <path-to.tif>` instead of relying on a baked-in filename.

**Other repository items**
- `CoordinateJSONs/`: GeoJSON inputs used by the attack. The current `Attack.py` defaults point to `CoordinateJSONs/OpenAddress/US.geojson` and `CoordinateJSONs/Synthetic/US.geojson`.
- `AugmentedFiles/`: generated scratch directory for intermediate renderings, candidate files, cluster-type cache files, and run logs. Ignored by Git.
- `Results/`: generated final run outputs organized by test name and dataset. Ignored by Git.
- `osm_cache/`: generated tile cache used by `contextily`. Ignored by Git.

**Inputs - formats and expectations**
- Map images: PNG or JPEG files containing red dot overlays on map tiles. PNG runs expect exact red `(255, 0, 0)`; JPEG runs use a red-dominant detector because compression changes the dot colors.
- GeoJSON: a `FeatureCollection` of `Point` features. Coordinates are read as `[longitude, latitude]`.
- Map bounds: the attack uses linear pixel-to-lat/lon conversion, so `PIXEL_SIZE`, `MIN_LON`, `MAX_LAT`, `WIDTH_PX`, and `HEIGHT_PX` must match the rendered map.
- Dataset/evaluation split: `RUN_DATASET` controls which GeoJSON is rendered for the attack, and `EVAL_SOURCE_FILES` controls the ground-truth GeoJSON used for scoring.

**Outputs - what you'll get**
- `Results/<TEST_NAME>/<dataset>/dot_center_results*.txt`: compact per-dot output containing pixel centers and converted latitude/longitude values.
- `Results/<TEST_NAME>/<dataset>/summary_results*.txt`: final summary metrics for the modified method and the baselines.
- `Results/<TEST_NAME>/<dataset>/descent_trace*.csv`: compact per-step descent diagnostics. Check `avg_geo_m`, `median_geo_m`, `moved_count`, and `avg_objective_gain` to see whether descent is improving centers or staying put.
- `Results/<TEST_NAME>/<dataset>/geo_error_histogram_boxplot*.<png|jpeg>` and `geo_error_boxplot*.pdf`: plots of modified-method geodesic errors.
- `AugmentedFiles/<TEST_NAME>/<dataset>/`: temporary candidate GeoJSONs/images, diagnostics, background-reference renders, manual cluster notes, and verbose run logs.

**Quick run**
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Run the current attack:

  ```bash
  python Attack.py
  ```

**Useful flags**
- `--dot-shape triangle` or `--dot-shape pentagon` to change the rendered dot marker.
- `--dot-shape 3,0,0` to pass an explicit Matplotlib marker tuple.
- `--log-level summary` keeps the console/run log short, while `--log-level verbose` restores the detailed progress output.
- `--shape-offset-px DX DY` applies an explicit pixel correction to detected marker centers for marker-anchor experiments.
- `--calibrate-shape-offset` estimates a triangle offset from the evaluation truth for diagnostics; do not use it for paper attack results.
- `--use-geometric-circle-init` starts isolated circle-dot runs from the refined geometric estimator. Leave this off when reproducing the original modified-method descent behavior.
- `--image-format jpeg --jpeg-quality 80` runs the GeoPandas-rendered JPEG attack path. PNG remains the default and most stable path.

**How to generate a map with GeoPandas (quick)**
- `Attack.py` already contains the map-generation helper used by the attack.
- If `REGENERATE_BASE_MAP = True`, the script will rerender the base map from the configured `JSON_FILE`.
- If `REGENERATE_BASE_MAP = False`, the script reuses the existing rendered map in `Results/<TEST_NAME>/<dataset>/` unless that file is missing.
- The rendered base map is saved as `map_<TEST_NAME>_<dataset>.png` for PNG runs or `.jpeg` for JPEG runs inside the corresponding results directory.

**Defense tool**
- `Defense.py` quantizes GeoJSON coordinates using local population density from a GeoTIFF raster and a `k`-based decimal truncation rule.
- Required inputs:
  - `--input`: input point GeoJSON. Defaults to `CoordinateJSONs/Synthetic/US.geojson`.
  - `--raster`: required population raster GeoTIFF (`.tif` or `.tiff`). This is not bundled in the repo.
  - `--output`: output GeoJSON. Defaults to `quantized.geojson`.
- Population rasters can be downloaded from the WorldPop population directory: https://data.worldpop.org/GIS/Population/
- Pick a raster that covers the coordinates in your input GeoJSON. For national US runs, use a US or global population GeoTIFF rather than a raster for another country/region.
- Basic example:

  ```bash
  python Defense.py --input CoordinateJSONs/Synthetic/US.geojson --raster "C:\path\to\worldpop_population.tif" --output quantized.geojson
  ```
- Example with defense parameters:

  ```bash
  python Defense.py --input CoordinateJSONs/OpenAddress/US.geojson --raster "C:\path\to\worldpop_population.tif" --output quantized_openaddresses.geojson --k 40 --radius-m 1000 --max-decimals 6
  ```
- `--k`: target minimum expected population per truncated coordinate cell. Larger values usually keep fewer decimal places.
- `--radius-m`: radius used to estimate local population density around each point. Default is `1000`.
- `--max-decimals`: maximum coordinate precision the defense is allowed to preserve. Default is `6`.

Notes:
- Ensure the GeoJSON uses WGS84 coordinates (`[lon, lat]`).
- If you change the dataset, bounds, zoom, image size, or dot size, rerender the base map before trusting attack results.

**Usage Framework (3-step)**
1) Configure `Attack.py`
  - Set `RUN_DATASET`, `EVAL_SOURCE_FILES`, `TEST_NAME`, and the map bounds / rendering constants.
  - Choose `CLUSTER_SIZE_MODE = "estimate"` for automatic blob-size estimation or `"manual"` if you want to label blob multiplicities yourself.

2) Run the method
  - Execute `python Attack.py`.
  - The script processes each dataset listed in `RUN_DATASET`.

3) Inspect results
  - Check `Results/<TEST_NAME>/<dataset>/dot_center_results*.txt` and `summary_results*.txt` first.
  - Check `Results/<TEST_NAME>/<dataset>/descent_trace*.csv` when diagnosing unexpectedly high error or whether descent is moving in the right direction.
  - Ignore `AugmentedFiles/...` unless you need debug visuals, candidate renders, or the detailed run log.

**Installation (minimal)**
- `pip`:

  ```bash
  pip install -r requirements.txt
  ```

# AutoLocate

This repository contains research code for recovering high-precision point locations from dot maps. The current main entry point is `Attack.py`, which renders or reuses a configured map, detects red dot clusters, refines their centers with perceptual descent, and converts recovered pixel centers back to geographic coordinates.

**Quick Summary**:
- **Purpose**: Recover geographic point locations from rendered dot maps by estimating dot centers at subpixel precision and mapping them back to latitude/longitude.
- **Main script**: `Attack.py`
- **Final outputs**: `Results/<TEST_NAME>/<dataset>/`
- **Intermediate artifacts**: `AugmentedFiles/<TEST_NAME>/<dataset>/`

**Requirements**:
- **Python**: 3.8+ recommended.
- **Current attack dependencies**: install from `requirements.txt`
  - `Pillow`, `scipy`, `geopy`, `numpy`, `pyproj`, `geopandas`, `matplotlib`, `contextily`, `xyzservices`
- Older helper/baseline scripts may need extra geospatial packages depending on your environment.

Note: `Attack.py` no longer uses the old `1` / `2` menu flow. A single run now ensures the base map exists (or regenerates it when `REGENERATE_BASE_MAP = True`) and then runs the attack.

**What each script does**
- `Attack.py`: Current cleaned attack pipeline for connected dot clusters.
  - Inputs: edit the top-level constants in the file, especially `RUN_DATASET`, `EVAL_SOURCE_FILES`, `TEST_NAME`, `CLUSTER_TYPE`, `CLUSTER_SIZE_MODE`, `BG_MODE`, `REGENERATE_BASE_MAP`, rendering settings, optimization settings, and map bounds.
  - Outputs: final artifacts in `Results/<TEST_NAME>/<dataset>/`, including the rendered map, rounded evaluation GeoJSON, per-dot results, summary metrics, and geodesic error plots.
  - Intermediate artifacts: candidate GeoJSONs and rendered candidate images, `BoundaryPixels*.png`, background-reference images, `manual_dot_queries*.txt`, `cluster_size*.png`, and `run_log*.txt` in `AugmentedFiles/<TEST_NAME>/<dataset>/`.
  - Notes: `CLUSTER_SIZE_MODE = "estimate"` estimates dots per red blob automatically. `CLUSTER_SIZE_MODE = "manual"` prompts for cluster sizes and caches them in `AugmentedFiles/cluster_types/<CLUSTER_TYPE>.geojson`.

- `UncleanedAttack.py`: Raw reference version of the attack logic. Useful for comparing behavior against the cleaned script.

- `Geometric-and-Naive.py`: Baseline detector that estimates red dot centers directly from an input map image without the full perceptual-descent loop.

- `Perceptual-Descent-Connected.py` and `Perceptual-Descent-Nonconnected.py`: Older perceptual-descent variants kept in the repository for comparison and experimentation.

**Other repository items**
- `CoordinateJSONs/`: GeoJSON inputs used by the attack. The current `Attack.py` defaults point to `CoordinateJSONs/OpenAddress/US.geojson` and `CoordinateJSONs/Synthetic/US.geojson`.
- `AugmentedFiles/`: scratch directory for intermediate renderings, candidate files, cluster-type cache files, and run logs.
- `Results/`: final run outputs organized by test name and dataset.
- `Media-Misc/`: example map images and related assets from earlier experiments.
- `osm_cache/`: tile cache used by `contextily`.

**Inputs - formats and expectations**
- Map images: PNG files containing red dot overlays on map tiles. The scripts expect the overlay color to be exact red `(255, 0, 0)` unless you change `DOT_COLOR`.
- GeoJSON: a `FeatureCollection` of `Point` features. Coordinates are read as `[longitude, latitude]`.
- Map bounds: the attack uses linear pixel-to-lat/lon conversion, so `PIXEL_SIZE`, `MIN_LON`, `MAX_LAT`, `WIDTH_PX`, and `HEIGHT_PX` must match the rendered map.
- Dataset/evaluation split: `RUN_DATASET` controls which GeoJSON is rendered for the attack, and `EVAL_SOURCE_FILES` controls the ground-truth GeoJSON used for scoring.

**Outputs - what you'll get**
- `Results/<TEST_NAME>/<dataset>/dot_center_results*.txt`: compact per-dot output containing pixel centers and converted latitude/longitude values.
- `Results/<TEST_NAME>/<dataset>/summary_results*.txt`: final summary metrics for the modified method and the baselines.
- `Results/<TEST_NAME>/<dataset>/geo_error_histogram_boxplot*.png` and `geo_error_boxplot*.pdf`: plots of modified-method geodesic errors.
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

**How to generate a map with GeoPandas (quick)**
- `Attack.py` already contains the map-generation helper used by the attack.
- If `REGENERATE_BASE_MAP = True`, the script will rerender the base map from the configured `JSON_FILE`.
- If `REGENERATE_BASE_MAP = False`, the script reuses the existing rendered map in `Results/<TEST_NAME>/<dataset>/` unless that file is missing.
- The rendered base map is saved as `map_<TEST_NAME>_<dataset>.png` inside the corresponding results directory.

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
  - Ignore `AugmentedFiles/...` unless you need debug visuals, candidate renders, or the detailed run log.

**Installation (minimal)**
- `pip`:

  ```bash
  pip install -r requirements.txt
  ```

# AutoLocate

This repository contains research code used to detect and recover high-precision point locations from dot maps (dot overlays on map images). 

**Quick Summary**:
- **Purpose**: Extract pixel centers of red dots from map images and convert those pixel locations back to geographic coordinates; includes a geometric/naive detector and two perceptual-descent implementations (connected and non-connected dot patterns).

**Requirements**:
- **Python**: 3.8+ recommended.
- **Core packages**: `Pillow`, `numpy`, `scipy`, `geopy`, `matplotlib`, `xyzservices`, `shapely`, `geopandas`, `pyproj`, `contextily`, `cartopy` (used in `Geometric-and-Naive.py`).
- On Windows, installing `geopandas`, `cartopy` and `contextily` is easiest via conda. See "Installation" below.
# AutoLocate

This README focuses on what each file in the repository does, what inputs the scripts expect, and what outputs they produce. Minimal setup notes are at the end.

Note: the interactive menu/interface used by the perceptual-descent scripts (the prompt where you press `1` to generate a map or `2` to run prediction) is printed to standard output. When the repository helper runs the scripts it often redirects stdout to `dot_center_results.txt`, so you may find the menu and runtime logs recorded in that file.

**What each script does**
- `Geometric-and-Naive.py`: Detects red dot clusters in an input map image and estimates each dot's pixel center using 
  geometric and pre-selected naive-method
  - Inputs: edit the top-level constants in the file — `FILENAME` (map image path), `COORDJSON` (GeoJSON with true points, optional for evaluation), and the map-to-lat/lon constants (`pixel_size`, `min_lon`, `max_lat`, `width_px`, `height_px`) when using different maps.
  - Outputs: Lists of dot center recovery errors (geodesic, pixel, lat/lon errors) in text document`dot_center_results.txt` (detection results, pixel and geodesic errors when `COORDJSON` is present) and console output. 

- `Perceptual-Descent-Connected.py`: Uses perceptual-descent to refine dot center estimates when dots form connected clusters (maps with clusters with multiple dots).
  - Inputs: edit the top-level constants in the file — `FILENAME` (map image path), `COORDJSON` (GeoJSON with true points, optional for evaluation), and the map-to-lat/lon constants (`pixel_size`, `min_lon`, `max_lat`, `width_px`, `height_px`) when using different maps
  - Outputs: `AugmentedMaps/` (created at runtime) containing temporary candidate GeoJSONs and rendered images used during optimization, `BoundaryPixels.png` (diagnostic), and `dot_center_results.txt` (final metrics).
  - Notes: Script is interactive (menu: generate starter map or predict coordinates). The algorithm clusters red pixels, fits circle models on cluster boundaries, and optionally performs iterative perceptual-descent by rendering candidate maps and minimizing pixel-level loss.

- `Perceptual-Descent-Nonconnected.py`: Same perceptual-descent approach adapted for isolated (non-connected) dots.
  - Inputs: `FILENAME` and `JSON_FILE` (defaults in the script). Map bounds constants apply.
  - Outputs: `AugmentedMaps/` with candidate GeoJSONs and images, `dot_center_results.txt` with final estimates and errors.
  - Notes: This variant uses a different initial-center extraction strategy suitable for isolated dots.

**Other repository items**
- `CoordinateJSONs/`: example GeoJSON files with point features. Use these with the perceptual-descent scripts to render ground-truth or candidate point sets.
- `Media-Misc/`: sample map images and associated world files. These are example inputs for `FILENAME`.

**Inputs — formats and expectations**
- Map images: PNG/JPEG files containing red dot overlays on map tiles. Scripts expect the overlay color to be roughly red `(255, 0, 0)` unless you change the `DOT_COLOR` constant.
- GeoJSON: a FeatureCollection of Point features. Scripts read coordinates as `[longitude, latitude]` pairs.
- Map bounds: the scripts use simple linear pixel↔lat/lon conversions. If you use your own map image you must provide matching values for `min_lon`, `max_lat`, `pixel_size`, and image resolution (`width_px`, `height_px`).


**Outputs — what you'll get**
- `dot_center_results.txt`: primary results file. Contains detected dot pixel centers, converted lat/lon (if ground truth is available and matching is performed), summary error metrics, and runtime info. This is the main file to inspect for experiment results.
- Intermediate artifacts: `AugmentedMaps/` and files like `BoundaryPixels.png` are created during perceptual-descent for debugging/visualization only — they are optional and can be ignored for normal use.

**Quick run (minimal)**
- Detect dots with the geometric approach:

  ```bash
  python Geometric-and-Naive.py
  ```

- Run perceptual-descent (connected clusters):

  ```bash
  python Perceptual-Descent-Connected.py
  ```

  The script will prompt to either generate a starter map or run prediction on the `FILENAME` configured in the file.

**How to generate a map with GeoPandas (quick)**
- The perceptual-descent scripts include a small map-generation helper that renders a GeoJSON of points onto an OSM basemap using GeoPandas.
- Quick steps:
  1. Open `Perceptual-Descent-Connected.py` (or `Perceptual-Descent-Nonconnected.py`).
  2. Run the script:

     ```bash
     python Perceptual-Descent-Connected.py
     ```

  3. When prompted, press `1` to "Generate a starter map". The script will render the GeoJSON set in `JSON_FILE` and save it to the `FILENAME` configured at the top of that script (default: `map.png`).

Notes:
- Ensure the GeoJSON uses WGS84 coordinates (`[lon, lat]`) and that `JSON_FILE` points to the file you want to render (example GeoJSONs are in `CoordinateJSONs/`).
- If you prefer programmatic control, you can render GeoJSONs with GeoPandas yourself (read the GeoJSON into a GeoDataFrame, plot it, and call `contextily.add_basemap`). The scripts' built-in option is the quickest path for experiments in this repo.

**Usage Framework (3-step)**
1) Generate map
  - Use the perceptual-descent script menu to generate a starter map (open `Perceptual-Descent-Connected.py` and input `1`), or render your own with GeoPandas/contextily. The generated map is saved to the `FILENAME` variable configured in the script (default: `map.png`).

2) Prepare GeoJSON
  - Prepare a GeoJSON FeatureCollection of Point features in WGS84 (`[lon, lat]`). You can reuse examples in `CoordinateJSONs/` (e.g., `CoordinateJSONs/Austin.geojson`).
  - Place your GeoJSON path into the script's `JSON_FILE` variable or pass it via the script's helper (the scripts read `JSON_FILE`).

3) Run the method
  - From the perceptual-descent script choose the `2) Predict coordinates` option to run the detection/refinement pipeline on the `FILENAME` map. Alternatively run `python Geometric-and-Naive.py` to use the geometric/naive detector.

Inspect results
 - Primary result: `dot_center_results.txt` — contains detected pixel centers, converted lat/lon (if ground truth available), summary error metrics, and runtime info.
 - Ignore intermediate `AugmentedMaps/` files unless you need debug visuals.



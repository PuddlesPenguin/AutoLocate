# AutoLocate

This repository contains research code used to detect and recover high-precision point locations from dot maps (dot overlays on map images). The implementations in this repo match methods described in the supplied paper: `_S_P26_submission__Privacy_Leakage_from_a_Thousand_Words__Millipixel_Location_Recovery_from_Dot_Maps.pdf` (see repository root).

**Quick Summary**:
- **Purpose**: Extract pixel centers of red dots from map images and convert those pixel locations back to geographic coordinates; includes a geometric/naive detector and two perceptual-descent implementations (connected and non-connected dot patterns).

**Requirements**:
- **Python**: 3.8+ recommended.
- **Core packages**: `Pillow`, `numpy`, `scipy`, `geopy`, `matplotlib`, `xyzservices`, `shapely`, `geopandas`, `pyproj`, `contextily`, `cartopy` (used in `Geometric-and-Naive.py`).
- On Windows, installing `geopandas`, `cartopy` and `contextily` is easiest via conda. See "Installation" below.

**Installation**
- Using pip (may fail for some geospatial packages on Windows):

  ```bash
  python -m pip install --upgrade pip
  python -m pip install pillow numpy scipy geopy matplotlib xyzservices shapely pyproj
  python -m pip install geopandas contextily cartopy
  ```

- Using conda (recommended on Windows):

  ```bash
  conda create -n autolocate python=3.9 -y
  conda activate autolocate
  conda install -c conda-forge geopandas contextily cartopy pyproj geopy shapely xyzservices matplotlib pillow scipy -y
  python -m pip install numpy
  ```

**Repository layout**
- `Geometric-and-Naive.py`: Geometric and naive dot-center detection, pixel↔lat/lon conversion, and matching to ground truth.
- `Perceptual-Descent-Connected.py`: Perceptual-descent method for maps where dots form connected clusters.
- `Perceptual-Descent-Nonconnected.py`: Perceptual-descent method for maps with non-connected (isolated) dots.
- `Perceptual-Descent-Manual`: (folder) appears to contain manual/auxiliary files for perceptual descent.
- `CoordinateJSONs/`: sample GeoJSONs used to generate maps (e.g. `Austin.geojson`, `Ohio.geojson`, `Non-Connected-US.geojson`).
- `Media-Misc/`: map image files and pgw/jgw world files used by scripts.
- `_S_P26_submission__...pdf`: the paper describing the methods implemented here.

**How to use — quick examples**
Notes: most scripts use top-level constants to configure filenames, coordinates, and map bounds. Edit the constants at the top of a script (e.g., `FILENAME`, `JSON_FILE`) before running, or adapt them into CLI wrappers.

- Geometric/Naive detector

  ```bash
  python Geometric-and-Naive.py
  ```

  - Configure at top of file: `FILENAME` (map image), `COORDJSON` (GeoJSON with true points), and the pixel/lat-lon bounding constants (`pixel_size`, `min_lon`, `max_lat`, `width_px`, `height_px`) if you use different maps.
  - Output: the script writes `dot_center_results.txt` with pixel and geographic errors and saves runtime info.

- Perceptual Descent (interactive choice in script)

  ```bash
  python Perceptual-Descent-Connected.py
  # or
  python Perceptual-Descent-Nonconnected.py
  ```

  - When the script runs it prints a simple menu:
    1) Generate a starter map — creates a georeferenced image from the GeoJSON in `JSON_FILE`.
    2) Predict coordinates — runs the perceptual-descent procedure on `FILENAME`.
  - Configure at top of file: `FILENAME`, `JSON_FILE`, and bounding constants (same as above).
  - Outputs and artifacts: generated small images (`left_img.png`, `right_img.png`, `up_img.png`, `down_img.png`, `_no_change.geojson` etc.), `BoundaryPixels.png` (connected variant), and `dot_center_results.txt`.

**Configuration notes**
- Both perceptual-descent and geometric scripts assume a fixed mapping from pixel coordinates to lat/lon — set these constants at the top of the script to your map's projection and resolution:
  - `pixel_size`, `width_px`, `height_px`, `min_lon`, `max_lat` (and derived `min_lat`, `max_lon`).
- Dot color and tolerances are defined by `DOT_COLOR`, `DOT_COLOR_RGB`, and `COLOR_THRESHOLD` in the respective files — change these if your dots use a different color or compression changes color values (JPEG).

**Data and example files**
- Use the GeoJSON examples in `CoordinateJSONs/` to generate maps. The `generate_map(...)` functions in the perceptual-descent scripts will render GeoPandas features onto an OSM basemap and save to the configured `FILENAME`.
- `Media-Misc/` contains sample map images and pgw/jgw world files used for tests and visualization.

**Notes about the algorithm & paper**
- The supplied PDF `_S_P26_submission__Privacy_Leakage_from_a_Thousand_Words__Millipixel_Location_Recovery_from_Dot_Maps.pdf` documents the threat model and recovery methods (millipixel recovery from dot overlays). The code in this repo implements:
  - Naive pixel-averaging and a weighted geometric center estimator (`Geometric-and-Naive.py`).
  - Perceptual-descent: iteratively rendering candidate point sets and comparing rendered images to the target map to minimize a pixel-level loss (`Perceptual-Descent-Connected.py` and `Perceptual-Descent-Nonconnected.py`).
---
Generated by a repository review of the code and the provided paper file in the repository root.
# AutoLocate
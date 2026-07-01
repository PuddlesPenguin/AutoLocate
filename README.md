# AutoLocate

AutoLocate recovers millipixel-precision latitude/longitude of locations from rasterized **dot maps** by
reverse-engineering the anti-aliasing artifacts produced when dots are rendered, showing that
published dot maps can leak individual locations even on a small-scale map (e.g., the US). This repository contains the attack, its evaluation, and a **standalone** privacy risk assessment tool that recommends a *k*-anonymous coordinate quantization level from local population density.

## Setup

```bash
pip install -r requirements.txt   # Python 3.8+
```

## Attack

Run AutoLocate (defaults to the national-scale US maps):

```bash
python attack.py                                    # all datasets
python attack.py --datasets Synthetic               # one dataset
python attack.py --test-name jpeg_q80 --image-format jpeg --jpeg-quality 80
```

Key flags: `--datasets {OpenAddresses,Synthetic}`, `--test-name`,
`--dot-shape {circle,triangle,pentagon}`, `--dot-radius-mm`, `--image-format {png,jpeg}`.

## Evaluation

Every run scores recovered points against the input coordinates (ground truth) and writes to
`Results/<test-name>/<dataset>/`:

- `summary_results_*.txt` — mean / median / p75 / p90 geodesic error (meters)
- `dot_center_results_*.txt` — recovered coordinate per point
- `geo_error_boxplot_*.pdf` — error distribution
- `descent_trace_*.csv` — per-iteration optimization trace

Intermediate renders go to `AugmentedFiles/`.

## Assessment tool

`assessment_tool.py` is a standalone tool that researchers run on their dot maps before publishing. Given coordinates, a population layer, and a target *k*, it selects the finest coordinate precision whose truncated cell is still expected to hold at least *k* people, then writes the quantized coordinates.

```bash
python assessment_tool.py \
  --input CoordinateJSONs/OpenAddress/US.geojson \
  --raster /path/to/worldpop.tif \
  --output quantized.geojson \
  --k 40
```

`--raster` is a required population GeoTIFF, e.g. WorldPop
(https://data.worldpop.org/GIS/Population/). For each point it reports the local density, decimals kept, expected people in the cell, and the quantized coordinate.

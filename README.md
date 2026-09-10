# AutoLocate

Code and data for [**Privacy Leakage from a Thousand Words: Millipixel Location Recovery from Dot Maps**](https://arxiv.org/abs/2609.07623), accepted to ACM CCS 2026.

AutoLocate is an automated framework for recovering the geographic coordinates represented by dots in a rendered map. It exploits anti-aliasing artifacts introduced during map rendering and treats recovery as a black-box optimization problem. The paper shows that these artifacts can leak sub-pixel location information, and introduces mitigation strategies plus a privacy-risk assessment tool for map publishers.

The public artifact deliberately separates three use cases:

- **Attack:** a map image goes in; predicted locations come out.
- **Assessment:** coordinates and a population raster go in; density-adaptive coordinate recommendations come out.
- **Evaluation:** ground truth and attack predictions go in; complete-distribution accuracy metrics come out.

The attack never needs ground-truth locations and does not calculate accuracy, prune outliers, or generate plots.

## Start here

Choose the workflow you need:

| Goal | Entry point | Input | Output |
| --- | --- | --- | --- |
| Recover locations from a rendered dot map | [Run the attack](#run-the-attack) | Map image and georeferencing parameters | Point GeoJSON |
| Measure recovery error | [Evaluate recovered locations](#evaluate-recovered-locations) | Truth and recovered Point GeoJSON | JSON metrics |
| Recommend safer coordinate precision | [Run the assessment tool](#run-the-assessment-tool) | Point GeoJSON and WorldPop GeoTIFF | Assessed Point GeoJSON |
| Browse bundled fixtures | [Data](#data) | Synthetic and OpenAddresses-derived samples | - |

The three commands are independent: evaluation does not run the attack, and assessment does not alter the input data.

## Project structure

```text
attack/                 Map-to-location attack implementation
assessment/             Standalone privacy assessment/quantization tool
evaluation/             Standalone all-point evaluation
CoordinateJSONs/        Small research datasets and overlap fixtures
attack.py               Attack command-line entry point
assessment_tool.py      Assessment command-line entry point
requirements.txt        Python dependencies
```

Local experiment matrices, job logs, caches, diagnostic workbooks, source rasters, and intermediate renders are excluded from the public surface.

## Setup

AutoLocate requires Python 3.10 or newer.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Map rendering uses web map tiles. A network connection is required on the first run; Contextily caches downloaded tiles for later runs.

## Run the attack

The public attack contract is:

```text
rendered map image + map bounds/style -> recovered Point GeoJSON
```

Example for a georeferenced PNG:

```bash
python attack.py \
  --input-image path/to/map.png \
  --output recovered_locations.geojson \
  --width-px 2284 \
  --height-px 1424 \
  --min-lon -126.17658145147593 \
  --max-lat 58.62037301762128 \
  --pixel-size 0.02587884152408056 \
  --dot-color 255 0 0 \
  --dot-radius-mm 2 \
  --dot-shape circle
```

The normal run writes exactly one result: the file passed to `--output`. Candidate renders are created in a temporary directory and deleted when the run finishes. Use `--save-debug-artifacts` only when developing the algorithm; it preserves intermediate images, candidate GeoJSON files, and the descent trace under `AugmentedFiles/`.

For a bundled end-to-end demonstration that first renders the included source data:

```bash
python attack.py --datasets Synthetic --test-name demo
```

This writes `Results/demo/Synthetic/recovered_locations.geojson`. The generated target image and optimizer work files remain temporary.

### Attack options

Run `python attack.py --help` for the parser-generated list. The complete reference is below.

| Option | Purpose |
| --- | --- |
| `--datasets [Synthetic] [OpenAddresses]` | Select bundled datasets. Omit to run both. Dataset mode renders source points first. |
| `--input-image PATH` | Existing rendered dot map. Source coordinates are not read. |
| `--output PATH` | Output Point GeoJSON. Default: `Results/<run>/<dataset>/recovered_locations.geojson`. |
| `--test-name` | Run name under `Results/` when `--output` is omitted. Default: `192dpiSatelliteBG-US`. |
| `--cluster-type` | Cluster initialization cache label. Default: `new`. |
| `--cluster-size-mode {estimate,manual}` | Estimate overlap sizes or request manual labels. Default: `estimate`. |
| `--bg-mode` | Enable known-background mode. Without it, unknown-background estimation is used. |
| `--regenerate-base-map` / `--no-regenerate-base-map` | Re-render or reuse the bundled target map. Default: regenerate. |
| `--min-lon VALUE`, `--max-lat VALUE`, `--pixel-size VALUE` | Geographic transform for pixels. Defaults: `-126.17658145147593`, `58.62037301762128`, `0.02587884152408056`. |
| `--width-px INTEGER`, `--height-px INTEGER` | Expected image dimensions. Defaults: `2284`, `1424`. |
| `--dot-color R G B` | RGB dot color. Default: `255 0 0`. |
| `--dot-shape VALUE` | `circle`, `triangle`, `pentagon`, or a Matplotlib marker tuple. Default: `circle`. |
| `--dot-radius-mm` | Rendered marker radius in millimeters. Default: `2`. |
| `--map-zoom` | Web-map tile zoom. Default: `5`. |
| `--max-iter` | Maximum cluster-center initialization iterations. Default: `100`. |
| `--tol` | Initialization convergence tolerance in pixels. Default: `0.01`. |
| `--initial-step-size` | First descent perturbation size in pixels. Default: `0.5`. |
| `--step-divisor` | Divisor applied after each descent step. Default: `1.2`. |
| `--min-step-size` | Stop descent below this pixel step. Default: `0.0001`. |
| `--log-level {summary,verbose}` | Progress detail. Default: `summary`. |
| `--seed` | Reproducible overlapping-dot initialization. Default: `0`. |
| `--quiet` | Suppress normal progress messages. |
| `--save-debug-artifacts` | Preserve intermediate renders and traces under `AugmentedFiles/`. Off by default. |
| `--image-format {png,jpeg}` | Candidate render format. Default: `png`; JPEG enables compressed-dot detection. |
| `--jpeg-quality` | JPEG quality from 1-100. Default: `80`. |
| `--shape-offset-px DX DY` | Fixed center offset for marker-anchor experiments. Default: `0 0`. |
| `--eval-decimals` | Legacy compatibility precision option. Default: `6`. |
| `--calibrate-shape-offset` | Legacy diagnostic calibration option; hidden from normal help. |
| `--use-geometric-circle-init` | Deprecated compatibility flag; isolated circle runs already use the refined initializer. |

For `--input-image`, provide `--output`, `--width-px`, `--height-px`, `--min-lon`, `--max-lat`, and `--pixel-size` as well. The attack then reads only the image and those georeferencing parameters.

## Evaluate recovered locations

Evaluation is a separate command and never runs as part of the attack:

```bash
python -m evaluation \
  --truth CoordinateJSONs/Synthetic/US.geojson \
  --recovered recovered_locations.geojson \
  --output metrics.json
```

The evaluator greedily matches each ground-truth point to the nearest unused prediction, then reports mean, sample standard deviation, minimum, median, p25/p75/p90, and maximum geodesic error. Every matched point is included; `outlier_pruning` is explicitly recorded as `none`.

Evaluation options:

| Option | Required | Purpose |
| --- | --- | --- |
| `--truth PATH` | Yes | Ground-truth Point GeoJSON. |
| `--recovered PATH` | Yes | Recovered Point GeoJSON from the attack. |
| `--output PATH` | No | Also write the printed metrics JSON to this path. |

## Run the assessment tool

The assessment tool is independent of the attack. It recommends the finest decimal precision whose estimated population cell contains at least `k` people.

Download an appropriate WorldPop GeoTIFF, then run:

```bash
python assessment_tool.py \
  --input CoordinateJSONs/Synthetic/US.geojson \
  --raster path/to/worldpop_population.tif \
  --output assessed_locations.geojson \
  --k 40 \
  --radius-m 1000 \
  --max-decimals 6
```

Normal output is one concise completion line and the assessed GeoJSON. Use `--verbose` only for per-point diagnostics. Download a population-count GeoTIFF from the official [WorldPop data catalog](https://www.worldpop.org/datacatalog/); the [population-count catalog](https://hub.worldpop.org/geodata/listing?id=29) lists available country, year, and resolution choices. WorldPop rasters are external and are not committed because they are large.

Assessment options:

| Option | Required | Default | Purpose |
| --- | --- | --- | --- |
| `--input PATH` | No | `CoordinateJSONs/Synthetic/US.geojson` | Input Point GeoJSON. |
| `--raster PATH` | Yes | - | Population-count GeoTIFF covering the input points. |
| `--output PATH` | No | `quantized.geojson` | Assessed Point GeoJSON. |
| `--k INTEGER` | No | `40` | Minimum expected people per retained decimal cell. |
| `--radius-m METERS` | No | `1000` | Radius used to estimate local population density. |
| `--max-decimals INTEGER` | No | `6` | Maximum coordinate precision to preserve. |
| `--verbose` | No | Off | Print one diagnostic line per input point. |

## Data

`CoordinateJSONs/` contains the small Point GeoJSON files used by the attack, evaluation, and assessment workflows.

| Path | Purpose |
| --- | --- |
| `Synthetic/` | Controlled point sets for reproducible experiments. |
| `OpenAddress/` | Samples derived from OpenAddresses source data. |
| `Synthetic-Clusters/` | Controlled overlap fixtures with two through five dots per cluster. |
| `Connected-US.geojson` | Connected/overlapping U.S. fixture. |

The default end-to-end example uses [`Synthetic/US.geojson`](CoordinateJSONs/Synthetic/US.geojson). The default OpenAddresses example uses [`OpenAddress/US.geojson`](CoordinateJSONs/OpenAddress/US.geojson). Each file is a GeoJSON `FeatureCollection` of `Point` features with coordinates in `[longitude, latitude]` order. OpenAddresses-derived fixtures retain their applicable source attribution and license requirements.

## Responsible use

AutoLocate demonstrates a location-privacy risk in rendered aggregate maps. Use it only on maps and data you are authorized to analyze. The assessment tool is provided to help map publishers test and reduce this leakage.

## Citation

If you use AutoLocate in your research, please cite:

```bibtex
@inproceedings{du2025systematic,
  title={Privacy Leakage from a Thousand Words: Millipixel Location Recovery from Dot Maps},
  author={Du, Yuntao and Pauskar, Tanishq and Wang, Hao and Su, Jing and Li, Ninghui},
  booktitle={Proceedings of the 33rd ACM SIGSAC Conference on Computer and Communications Security (CCS 2026)},
  year={2026}
}
```

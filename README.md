# AutoLocate

AutoLocate is research code for recovering high-precision point locations from rendered dot maps. The main entry point is `Attack.py`; the defense/quantization tool is `Defense.py`.

## Install

```bash
pip install -r requirements.txt
```

## Run the Attack

Default run:

```bash
python Attack.py
```

Run selected datasets:

```bash
python Attack.py --datasets Synthetic
python Attack.py --datasets OpenAddresses Synthetic
```

Common options:

```bash
python Attack.py --test-name baseline --dot-shape circle --dot-radius-mm 2
python Attack.py --test-name triangle_r2 --dot-shape triangle --dot-radius-mm 2
python Attack.py --test-name pentagon_r2 --dot-shape pentagon --dot-radius-mm 2
python Attack.py --test-name jpeg_q80 --image-format jpeg --jpeg-quality 80
```

Useful flags:

- `--datasets Synthetic` or `--datasets OpenAddresses Synthetic`
- `--test-name <name>`
- `--dot-shape circle|triangle|pentagon`
- `--dot-radius-mm <value>`
- `--image-format png|jpeg`
- `--jpeg-quality <1-100>`
- `--log-level summary|verbose`

Outputs are written to:

```text
Results/<TEST_NAME>/<dataset>/
```

Temporary attack artifacts are written to:

```text
AugmentedFiles/<TEST_NAME>/<dataset>/
```

Both folders are ignored by Git.

## Run the Defense

`Defense.py` quantizes point coordinates using local population density from an external population GeoTIFF.

Population rasters can be downloaded from WorldPop:

```text
https://data.worldpop.org/GIS/Population/
```

Basic example:

```bash
python Defense.py --input CoordinateJSONs/Synthetic/US.geojson --raster "C:\path\to\worldpop_population.tif" --output quantized.geojson
```

With parameters:

```bash
python Defense.py --input CoordinateJSONs/OpenAddress/US.geojson --raster "C:\path\to\worldpop_population.tif" --output quantized_openaddresses.geojson --k 40 --radius-m 1000 --max-decimals 6
```

Defense flags:

- `--input`: input point GeoJSON
- `--raster`: required population GeoTIFF (`.tif` or `.tiff`)
- `--output`: output GeoJSON
- `--k`: target minimum expected population per truncated coordinate cell
- `--radius-m`: radius for local density estimation
- `--max-decimals`: maximum coordinate precision to preserve

## Repository Layout

- `Attack.py`: attack pipeline
- `attack_parser.py`: CLI defaults and path configuration
- `attack_utils.py`: GeoJSON, rendering, and output helpers
- `Defense.py`: population-density quantization defense
- `CoordinateJSONs/`: input GeoJSON datasets
- `requirements.txt`: Python dependencies

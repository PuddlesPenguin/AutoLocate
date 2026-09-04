import argparse
import json
import math
from decimal import Decimal, ROUND_DOWN
from pathlib import Path


def get_local_population_density(lat, lon, raster_path, radius_m=1000):
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("Rasterio is required to read the population GeoTIFF; install requirements.txt.") from exc

    with rasterio.open(raster_path) as src:
        degree_buffer = radius_m / 111_000
        minx, maxx = lon - degree_buffer, lon + degree_buffer
        miny, maxy = lat - degree_buffer, lat + degree_buffer

        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(1, window=window)

        data = data.astype(float)
        data[data == src.nodata] = 0

        population = float(data.sum())
        area_km2 = math.pi * (radius_m/1000)**2
        density = population / area_km2
    return density

def truncate_to_decimals(value, decimals):
    """
    Truncate a coordinate toward zero at the requested decimal precision.
    """
    quantizer = Decimal("1").scaleb(-decimals)
    truncated = Decimal(str(value)).quantize(quantizer, rounding=ROUND_DOWN)
    return float(truncated)


def expected_people_in_decimal_cell(lat, density, decimals):
    """
    Estimate how many people fall inside a cell formed by truncating latitude
    and longitude to `decimals` decimal places.
    """
    if density <= 0:
        return 0.0

    step_deg = 10 ** (-decimals)
    lat_km = 111.0 * step_deg
    lon_km = 111.0 * max(abs(math.cos(math.radians(lat))), 1e-9) * step_deg
    area_km2 = lat_km * lon_km
    return density * area_km2


def choose_decimal_precision(lat, density, k, max_decimals=6):
    """
    Pick the finest decimal precision that still has an expected population of
    at least `k` inside the truncated cell.
    """
    for decimals in range(max_decimals, -1, -1):
        if expected_people_in_decimal_cell(lat, density, decimals) >= k:
            return decimals
    return 0

DEFAULT_INPUT_FILE = Path("CoordinateJSONs/Synthetic/US.geojson")
DEFAULT_RASTER_PATH = None
DEFAULT_OUTPUT_FILE = Path("quantized.geojson")


def resolve_raster_path(raster_path):
    raster_path = Path(raster_path).expanduser()
    if not raster_path.exists():
        raise FileNotFoundError(f"Population raster GeoTIFF not found: {raster_path}")
    return raster_path


def process_geojson_quantize(
    input_file,
    raster_path,
    output_file,
    k_protection=40,
    radius_m=1000,
    max_decimals=6,
    verbose=False,
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    quantized_features = []
    skipped = 0
    for feature in data["features"]:
        lon, lat = feature["geometry"]["coordinates"]

        try:
            density = get_local_population_density(lat, lon, raster_path, radius_m)
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"Skipping point {lat},{lon}: {e}")
            continue

        decimals = choose_decimal_precision(lat, density, k_protection, max_decimals=max_decimals)
        new_lat = truncate_to_decimals(lat, decimals)
        new_lon = truncate_to_decimals(lon, decimals)
        expected_people = expected_people_in_decimal_cell(lat, density, decimals)

        if verbose:
            print(
                f"({lat}, {lon}) -> ({new_lat}, {new_lon}); "
                f"density={density:.2f} people/km^2, decimals={decimals}, "
                f"expected_people={expected_people:.2f}"
            )

        quantized_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [new_lon, new_lat]},
            "properties": feature.get("properties", {})
        })

    quantized_data = {"type": "FeatureCollection", "features": quantized_features}
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(quantized_data, f, indent=2)
    return {"processed": len(quantized_features), "skipped": skipped, "output": str(output_file)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize coordinates using local population density and k-based decimal truncation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input GeoJSON file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--raster",
        type=Path,
        required=True,
        help=(
            "Population raster GeoTIFF. WorldPop downloads are available at "
            "https://data.worldpop.org/GIS/Population/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output GeoJSON file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=40,
        help="Target k value used to choose decimal precision (default: 40)",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=1000,
        help="Radius in meters used for local density estimation (default: 1000)",
    )
    parser.add_argument(
        "--max-decimals",
        type=int,
        default=6,
        help="Maximum number of decimals to preserve (default: 6)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one diagnostic line per input point.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.raster = resolve_raster_path(args.raster)
    result = process_geojson_quantize(
        args.input,
        args.raster,
        args.output,
        k_protection=args.k,
        radius_m=args.radius_m,
        max_decimals=args.max_decimals,
        verbose=args.verbose,
    )
    print(f"Wrote {result['processed']} assessed locations to {result['output']} ({result['skipped']} skipped).")


if __name__ == "__main__":
    main()

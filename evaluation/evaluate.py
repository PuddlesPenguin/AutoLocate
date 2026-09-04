"""Evaluate recovered locations against ground truth without outlier pruning."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from geopy.distance import geodesic as _geodesic
except ImportError:  # Keep the standalone evaluator usable in a bare Python environment.
    _geodesic = None


def distance_m(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    if _geodesic is not None:
        return float(_geodesic(point_a, point_b).meters)
    lat1, lon1 = map(math.radians, point_a)
    lat2, lon2 = map(math.radians, point_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_008.8 * math.asin(min(1.0, math.sqrt(haversine)))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def load_points(path: Path) -> list[tuple[float, float]]:
    """Return GeoJSON Point coordinates as ``(latitude, longitude)`` pairs."""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    points = []
    for feature in document.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") == "Point" and coordinates and len(coordinates) >= 2:
            lon, lat = coordinates[:2]
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                points.append((float(lat), float(lon)))
    return points


def greedy_match(
    truth: list[tuple[float, float]],
    recovered: list[tuple[float, float]],
) -> list[float]:
    """Match each truth point to the nearest unused recovered point."""
    available = set(range(len(recovered)))
    errors = []
    for true_point in truth:
        if not available:
            break
        recovered_index, error_m = min(
            ((index, distance_m(true_point, recovered[index])) for index in available),
            key=lambda pair: pair[1],
        )
        available.remove(recovered_index)
        errors.append(float(error_m))
    return errors


def summarize(errors: list[float], truth_count: int, recovered_count: int) -> dict:
    if not errors:
        return {
            "truth_count": truth_count,
            "recovered_count": recovered_count,
            "matched_count": 0,
            "error_m": None,
        }
    return {
        "truth_count": truth_count,
        "recovered_count": recovered_count,
        "matched_count": len(errors),
        "unmatched_truth_count": truth_count - len(errors),
        "unmatched_recovered_count": recovered_count - len(errors),
        "error_m": {
            "mean": statistics.fmean(errors),
            "sample_std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
            "min": min(errors),
            "p25": percentile(errors, 25),
            "median": statistics.median(errors),
            "p75": percentile(errors, 75),
            "p90": percentile(errors, 90),
            "max": max(errors),
        },
        "outlier_pruning": "none",
    }


def evaluate(truth_path: Path, recovered_path: Path) -> dict:
    truth = load_points(truth_path)
    recovered = load_points(recovered_path)
    return summarize(greedy_match(truth, recovered), len(truth), len(recovered))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="Ground-truth Point GeoJSON.")
    parser.add_argument("--recovered", type=Path, required=True, help="Recovered Point GeoJSON from the attack.")
    parser.add_argument("--output", type=Path, help="Optional JSON metrics file. Metrics are always printed to stdout.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metrics = evaluate(args.truth, args.recovered)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main()

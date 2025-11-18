import json
import os
import random
import math
from typing import Tuple, List

import requests
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union

# Script used to generate point clusters of fixed sizes

SEED = 1234
random.seed(SEED)

NUM_CLUSTERS = 25           # number of clusters
POINTS_PER_CLUSTER = 4    # exactly 4 dots per cluster
CLUSTER_SPREAD_KM = 30.0    # how far from the center we allow placement
MIN_SEPARATION_KM = 25.0    # minimum spacing between dots in a cluster
MIN_CLUSTER_SEPARATION_KM = 200.0  # minimum spacing between cluster centers

CONUS_ONLY = True
MIN_LAT, MAX_LAT = 24.5, 49.5
MIN_LON, MAX_LON = -124.8, -66.9

LOCAL_USA_GEOJSON = "usa.geojson"
FALLBACK_USA_RAW = (
    "https://raw.githubusercontent.com/johan/world.geo.json/"
    "master/countries/USA.geo.json"
)
OUTPUT_GEOJSON = "clusters_output.geojson"

def download_usa_geojson(dest: str):
    resp = requests.get(FALLBACK_USA_RAW, timeout=20)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)


def load_usa_polygon():
    if os.path.exists(LOCAL_USA_GEOJSON):
        with open(LOCAL_USA_GEOJSON, "r", encoding="utf-8") as f:
            gj = json.load(f)
    else:
        download_usa_geojson(LOCAL_USA_GEOJSON)
        with open(LOCAL_USA_GEOJSON, "r", encoding="utf-8") as f:
            gj = json.load(f)

    if gj.get("type") == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in gj["features"] if feat.get("geometry")]
        us_geom = unary_union(geoms)
    elif gj.get("type") == "Feature":
        us_geom = shape(gj["geometry"])
    else:
        us_geom = shape(gj)
    return us_geom


def destination_point(lat, lon, bearing_deg, distance_km):
    R = 6371.0088
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    d = distance_km / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lon2) + 540) % 360) - 180


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def generate_cluster(center, n, spread_km, min_sep_km, polygon):
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 10000:
        attempts += 1
        bearing = random.uniform(0, 360)
        dist = random.uniform(0, spread_km)
        lat2, lon2 = destination_point(center[0], center[1], bearing, dist)
        if not polygon.contains(Point(lon2, lat2)):
            continue
        # enforce minimum spacing
        too_close = any(haversine(lat2, lon2, p[0], p[1]) < min_sep_km for p in pts)
        if not too_close:
            pts.append((lat2, lon2))
    return pts


def main():
    us_geom = load_usa_polygon()
    if CONUS_ONLY:
        bound_rect = box(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)
        us_geom = us_geom.intersection(bound_rect)

    minx, miny, maxx, maxy = us_geom.bounds
    clusters = []

    for cid in range(NUM_CLUSTERS):
        # find cluster center far enough from existing clusters
        for _ in range(10000):
            lat = random.uniform(miny, maxy)
            lon = random.uniform(minx, maxx)
            if not us_geom.contains(Point(lon, lat)):
                continue
            # new check: enforce distance from all previous cluster centers
            if all(haversine(lat, lon, c[0], c[1]) >= MIN_CLUSTER_SEPARATION_KM for c, _ in clusters):
                center = (lat, lon)
                break
        else:
            raise RuntimeError(f"Failed to place cluster {cid} after many tries.")

        pts = generate_cluster(center, POINTS_PER_CLUSTER, CLUSTER_SPREAD_KM, MIN_SEPARATION_KM, us_geom)
        clusters.append((center, pts))

    features = []
    for cid, (center, pts) in enumerate(clusters):
        for midx, (plat, plon) in enumerate(pts):
            features.append({
                "type": "Feature",
                "properties": {"cluster_id": cid, "point_id": midx},
                "geometry": {"type": "Point", "coordinates": [plon, plat]}
            })
    geo = {"type": "FeatureCollection", "features": features}
    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)
    print(f"Wrote {OUTPUT_GEOJSON} with {len(features)} points.")


if __name__ == "__main__":
    main()

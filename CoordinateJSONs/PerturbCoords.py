import json
import random
import math
from pyproj import Geod

INPUT_FILE = "coords.geojson"
OUTPUT_FILE = "output.geojson"
PERTURB_DISTANCE = 100

geod = Geod(ellps="WGS84")

def perturb_point(lon, lat, dist):
    theta = random.random() * 2 * math.pi
    bearing = math.degrees(theta)

    new_lon, new_lat, _ = geod.fwd(lon, lat, bearing, dist)
    return new_lon, new_lat


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    for feature in data["features"]:
        geom = feature["geometry"]

        if geom["type"] == "Point":
            lon, lat = geom["coordinates"]
            geom["coordinates"] = list(
                perturb_point(lon, lat, PERTURB_DISTANCE)
            )

        elif geom["type"] == "MultiPoint":
            new_coords = []
            for lon, lat in geom["coordinates"]:
                new_coords.append(
                    list(perturb_point(lon, lat, PERTURB_DISTANCE))
                )
            geom["coordinates"] = new_coords

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved perturbed GeoJSON to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

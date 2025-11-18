from PIL import Image, ImageDraw  # noqa
from collections import deque, defaultdict
import random
import math
from scipy.optimize import brentq
import numpy as np  # noqa
import contextlib
import json
from geopy.distance import geodesic
import numpy as np
import sys
import os
import pyproj
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import xyzservices.providers as xyz
from shapely.geometry import box
from scipy.optimize import linear_sum_assignment
import xyzservices.providers as xyz
import time
from xyzservices import TileProvider
'''
 This script allows you to
 1) Generate GeoPandes Maps with intended parameters and Open Streetmap Background
    Adjust generate_map() method to change map
 2) Use Perceptual Descent Method on a map of choosing. Will print out results to dot_center_results.py
    For maps with no connected dots
'''

PIXELS_PER_DOT = 131
MAX_ITER = 100
TOL = 1e-2
RADIUS = 7.5
DOT_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)
FILENAME = "map.png" # Change to image destination
JSON_FILE = "coords.geojson" # Change to geojson file with coordinates

pixel_size = 0.02587884152408056 # Latitude / Pixel
width_px = 2284 # Pixels / Width
height_px = 1424 # Pixels / Height
min_lon = -126.17658145147592563 # Longitude Bound
max_lat = 58.62037301762128294 # Latitude Bound
max_lon = min_lon + pixel_size * width_px
min_lat = max_lat - pixel_size * height_px
dirs = [(-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),         ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)]


def safe_input():
    while True:
        line = input().strip()
        if line:
            return line
def generate_map(predicted_points: str, output_path: str):
    """
    Generate a stretched visual map from a GeoJSON of predicted points
    and save it to the specified output path.
    """
    import json, tempfile, os
    from shapely.geometry import mapping, Point

    os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
    print("Using PROJ database at:", os.environ["PROJ_LIB"])

    if predicted_points == "none":
        import shutil, tempfile, os

        src_path = JSON_FILE
        tmp_path = os.path.join(tempfile.gettempdir(), "dummy_points.geojson")

        shutil.copyfile(src_path, tmp_path)
        predicted_points = tmp_path

        print(f"Copied {src_path} to temporary file: {tmp_path}")

    gdf = gpd.read_file(predicted_points)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(epsg=4326)

    bounds_geom = box(min_lon, min_lat, max_lon, max_lat)
    minx, miny, maxx, maxy = bounds_geom.bounds

    #tile_source = xyz.CartoDB.PositronNoLabels
    tile_source = xyz.OpenStreetMap.Mapnik
    ctx.set_cache_dir("osm_cache")

    stretch_factor = 1
    fig_w_in = (width_px / 96) * stretch_factor
    fig_h_in = height_px / 96
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=96)

    mm_to_pt = 72 / 25.4
    marker_diameter_mm = 4.0
    marker_size_pts2 = (marker_diameter_mm * mm_to_pt) ** 2

    gdf.plot(
        ax=ax,
        color="#ff0000",
        markersize=marker_size_pts2,
        edgecolor="none",
        linewidth=0,
        alpha=0 if "dummy_points.geojson" in predicted_points else 1
    )


    ctx.add_basemap(
        ax,
        source=tile_source,
        crs="EPSG:4326",
        zoom=5,
        reset_extent=False
    )

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect(1 / stretch_factor)

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(output_path, dpi=96, bbox_inches=None, pad_inches=0, facecolor="none")
    plt.close(fig)
    print(f"Map saved to: {output_path}")
def lat_lon_to_pixel(lat, lon, img_width, img_height):
    x = (lon - min_lon) / (max_lon - min_lon) * img_width
    y = (max_lat - lat) / (max_lat - min_lat) * img_height
    return (x, y)
def pixel_to_lat_lon(x, y, img_width, img_height):
    lon = min_lon + (x / img_width) * (max_lon - min_lon)
    lat = max_lat - (y / img_height) * (max_lat - min_lat)
    return (lat, lon)
def color_distance(c1, c2):
    return (sum(abs(a - b) for a, b in zip(c1, c2)))
def region_area(r, theta):
    mid_x = 0.5 + r * math.cos(theta)
    mid_y = 0.5 + r * math.sin(theta)
    if mid_x >= 1 or mid_y >= 1:
        return 1
    if mid_x <= 0 or mid_y <= 0:
        return 0
    if r < 0:
        if mid_x ** 2 + (1 - mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.cos(theta)
        elif (1 - mid_x) ** 2 + (mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.sin(theta)
        else:
            d = 1/math.sqrt(2) * math.cos(math.pi/4 - theta)
            return 1/2 * 1/math.cos(theta) * 1/math.sin(theta) * (d + r) ** 2
    else:
        if mid_x ** 2 + (1 - mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.sin(theta)
        elif (1 - mid_x) ** 2 + (mid_y) ** 2 + r ** 2 < 1/2:
            return 1/2 + r / math.cos(theta)
        else:
            d = 1/math.sqrt(2) * math.cos(math.pi/4 - theta)
            return 1 - 1/2 * 1/math.cos(theta) * 1/math.sin(theta) * (d - r) ** 2
def find_r(theta, area):
    def objective(r):
        return region_area(r, theta) - area
    
    r_min = -10
    r_max = 10
    
    f_min = objective(r_min)
    f_max = objective(r_max)
    
    if f_min * f_max > 0:
        raise ValueError("No solution found within the range [-10, 10]. Ensure area is between 0 and 1 and theta is valid.")
    
    r_solution = brentq(objective, r_min, r_max)
    return r_solution
def generate_unique_color(i):
    random.seed(i)
    return tuple(random.randint(50, 255) for _ in range(3))
def get_blobs(image):
    pixels = image.load()
    width, height = image.size
    visited = [[False for _ in range(height)] for _ in range(width)]
    blobs = []
    for x in range(width):
        for y in range(height):
            if pixels[x,y] == DOT_COLOR and not visited[x][y]:
                queue = deque()
                queue.append((x, y))
                blob = []
                while queue:
                    cx, cy = queue.popleft()
                    if not (0 <= cx < width and 0 <= cy < height):
                        continue
                    if visited[cx][cy] or pixels[cx, cy] != (255, 0, 0):
                        continue
                    visited[cx][cy] = True
                    blob.append((cx, cy))
                    for dx, dy in dirs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                            queue.append((nx, ny))
                blobs.append(blob)
    return blobs
def create_geojson(latlon_list, output_path):
    """
    Creates a GeoJSON file from a list of (latitude, longitude) tuples.

    Parameters:
        latlon_list (list of (lat, lon)): List of geographic coordinates.
        output_path (str): Path to save the output .geojson file.
    """
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    for lat, lon in latlon_list:
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        }
        geojson_data["features"].append(feature)

    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
def get_true_points(filename=JSON_FILE):
    with open(filename, "r") as f:
        data = json.load(f)
    
    true_points = []
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords:
            true_points.append((coords[0], coords[1]))
    return true_points
def match_estimates(true_centers, estimated_centers):
    true_centers = np.array(true_centers)
    estimated_centers = np.array(estimated_centers)
    n_true = len(true_centers)
    n_est = len(estimated_centers)
    n = max(n_true, n_est)

    cost_matrix = np.full((n, n), fill_value=1e6)
    for i in range(n_true):
        for j in range(n_est):
            dx = true_centers[i][0] - estimated_centers[j][0]
            dy = true_centers[i][1] - estimated_centers[j][1]
            cost_matrix[i][j] = dx**2 + dy**2

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    errors = []
    matched = []
    for i, j in zip(row_ind, col_ind):
        if i < n_true and j < n_est:
            true = true_centers[i]
            est = estimated_centers[j]
            errors.append(((abs(true[0] - est[0])), (abs(true[1] - est[1]))))
            matched.append(tuple(est))
    return errors, matched
def main():
    print("Pick an option")
    print("1) Generate a starter map")
    print("2) Predict coordinates")
    x = int(safe_input())
    if x == 1:
        generate_map(JSON_FILE, "__________.png")
        generate_map("none", "__________bg.png")

    else:
        image = Image.open(FILENAME).convert("RGB")
        pixels = image.load()
        width, height = image.size
        original_image = Image.open(FILENAME).convert("RGB")
        original_pixels = original_image.load()
        blobs = get_blobs(image)
        print(blobs)
        def get_initial_centers():
            cluster_pixels = defaultdict(list)
            new_centers = []
            dirs = [(1,0),(-1,0),(0,1),(0,-1)]
            for blob_index, blob in enumerate(blobs):
                cid = blob_index * 1000
                blob_set = set(blob)
                boundary = set()
                for x, y in blob:
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) not in blob_set:
                            boundary.add((nx, ny))
                cluster_pixels[cid] = list(blob_set | boundary)
                avg_x = sum(x for x, y in blob) / len(blob)
                avg_y = sum(y for x, y in blob) / len(blob)
                new_centers.append((cid, (avg_x + 0.5, avg_y + 0.5)))
            return new_centers, cluster_pixels
        initial_centers, cluster_pixels = get_initial_centers()

        pairs = sorted(initial_centers, key=lambda t: (t[1][0], t[1][1]))
        prev_centers = [p[1] for p in pairs]
        keys = [p[0] for p in pairs]

        print(initial_centers)
        print(keys)
        
        def calc_error_for_dot(base, new):
            res = [0 for _ in range(len(keys))]
            for i in range(len(keys)):
                error = 0
                for x,y in cluster_pixels[keys[i]]:
                    error += abs(color_distance(base[x, y], new[x, y]))
                res[i] = error
            return res
                
        step_size = 0.05
        step_div = 1.2
        def print_error():
            estimated_centers = prev_centers
            png_path = FILENAME
            true_locations = get_true_points()
            true_centers = [lat_lon_to_pixel(loc[1], loc[0], width, height) for loc in true_locations]             
            good_errors, matched_good = match_estimates(true_centers, estimated_centers)
            true_latlons = [(loc[1], loc[0]) for loc in true_locations]
            good_pred_latlons = [pixel_to_lat_lon(x, y, width, height) for x, y in matched_good]
            good_geo_errors = [geodesic(true, pred).meters for true, pred in zip(true_latlons, good_pred_latlons)]

            def average_coord_error(true_coords, pred_coords):
                lat_errors = [abs(pred[0] - true[0]) for true, pred in zip(true_coords, pred_coords)]
                lon_errors = [abs(pred[1] - true[1]) for true, pred in zip(true_coords, pred_coords)]
                return sum(lat_errors) / len(lat_errors), sum(lon_errors) / len(lon_errors)

            def average_pixel_error(true_pixels, pred_pixels):
                x_errors = [abs(pred[0] - true[0]) for true, pred in zip(true_pixels, pred_pixels)]
                y_errors = [abs(pred[1] - true[1]) for true, pred in zip(true_pixels, pred_pixels)]
                return sum(x_errors) / len(x_errors), sum(y_errors) / len(y_errors)

            def average(lst):
                return sum(lst) / len(lst) if lst else 0.0

            good_avg_lat_error, good_avg_lon_error = average_coord_error(true_latlons, good_pred_latlons)
            good_avg_x_error, good_avg_y_error = average_pixel_error(true_centers, matched_good)
            good_geo_error = average(good_geo_errors)
            print(f"\nGOOD AVG PIXEL ERROR: x = {good_avg_x_error:.8f} px, y = {good_avg_y_error:.8f} px", flush=True)
        loss_avg = []
        while step_size > 0.000001:
            left = [(prev_centers[i][0] - step_size, prev_centers[i][1]) for i in range(len(keys))]
            left_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in left]
            create_geojson(left_coords, "_left.geojson")
            generate_map("_left.geojson", "left_img.png")
            left_pixels = Image.open("left_img.png").convert("RGB").load()
            left_err = calc_error_for_dot(original_pixels, left_pixels)
            
            right = [(prev_centers[i][0] + step_size, prev_centers[i][1]) for i in range(len(keys))]
            right_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in right]
            create_geojson(right_coords, "_right.geojson")
            generate_map("_right.geojson", "right_img.png")
            right_pixels = Image.open("right_img.png").convert("RGB").load()
            right_err = calc_error_for_dot(original_pixels, right_pixels)

            up = [(prev_centers[i][0], prev_centers[i][1] + step_size) for i in range(len(keys))]
            up_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in up]
            create_geojson(up_coords, "_up.geojson")
            generate_map("_up.geojson", "up_img.png")
            up_pixels = Image.open("up_img.png").convert("RGB").load()
            up_err = calc_error_for_dot(original_pixels, up_pixels)

            down = [(prev_centers[i][0], prev_centers[i][1] - step_size) for i in range(len(keys))]
            down_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in down]
            create_geojson(down_coords, "_down.geojson")
            generate_map("_down.geojson", "down_img.png")
            down_pixels = Image.open("down_img.png").convert("RGB").load()
            down_err = calc_error_for_dot(original_pixels, down_pixels)

            no_change = [(prev_centers[i][0], prev_centers[i][1]) for i in range(len(keys))]
            no_change_coords = [pixel_to_lat_lon(c[0], c[1], width, height) for c in no_change]
            create_geojson(no_change_coords, "_no_change.geojson")
            generate_map("_no_change.geojson", "no_change_img.png")
            no_change_pixels = Image.open("no_change_img.png").convert("RGB").load()
            no_change_err = calc_error_for_dot(original_pixels, no_change_pixels)
            loss_arr = []
            new_pix_coords = [(0, 0) for i in range(len(keys))]
            for i in range(len(keys)):
                loss_arr.append(no_change_err[i])
                min_err = min(left_err[i], right_err[i], up_err[i], down_err[i], no_change_err[i])
                print(min_err)
                print(f"{left_err[i]} {right_err[i]} {up_err[i]} {down_err[i]} {no_change_err[i]}")
                if left_err[i] == min_err:
                    print("Going left")
                    new_pix_coords[i] = left[i]
                elif right_err[i] == min_err:
                    print("Going right")
                    new_pix_coords[i] = right[i]
                elif up_err[i] == min_err:
                    print("Going up")
                    new_pix_coords[i] = up[i]
                elif down_err[i] == min_err:
                    print("Going down")
                    new_pix_coords[i] = down[i]
                else:
                    print("Staying same")
                    new_pix_coords[i] = no_change[i]
            loss_avg.append(round(sum(loss_arr)/len(loss_arr), 3))
            prev_centers = new_pix_coords
            print_error()
            step_size /= step_div
            print(step_size)
        print(f"LOSS ARRAY: {loss_avg}")
        print(prev_centers)
        estimated_centers = prev_centers
        png_path = FILENAME
        true_locations = get_true_points()
        true_centers = [lat_lon_to_pixel(loc[1], loc[0], width, height) for loc in true_locations]             
        good_errors, matched_good = match_estimates(true_centers, estimated_centers)
        true_latlons = [(loc[1], loc[0]) for loc in true_locations]
        good_pred_latlons = [pixel_to_lat_lon(x, y, width, height) for x, y in matched_good]
        good_geo_errors = [geodesic(true, pred).meters for true, pred in zip(true_latlons, good_pred_latlons)]

        def average_coord_error(true_coords, pred_coords):
            lat_errors = [abs(pred[0] - true[0]) for true, pred in zip(true_coords, pred_coords)]
            lon_errors = [abs(pred[1] - true[1]) for true, pred in zip(true_coords, pred_coords)]
            return sum(lat_errors) / len(lat_errors), sum(lon_errors) / len(lon_errors)

        def average_pixel_error(true_pixels, pred_pixels):
            x_errors = [abs(pred[0] - true[0]) for true, pred in zip(true_pixels, pred_pixels)]
            y_errors = [abs(pred[1] - true[1]) for true, pred in zip(true_pixels, pred_pixels)]
            return sum(x_errors) / len(x_errors), sum(y_errors) / len(y_errors)

        def average(lst):
            return sum(lst) / len(lst) if lst else 0.0

        good_avg_lat_error, good_avg_lon_error = average_coord_error(true_latlons, good_pred_latlons)
        good_avg_x_error, good_avg_y_error = average_pixel_error(true_centers, matched_good)
        good_geo_error = average(good_geo_errors)
        print(f"\nGOOD AVG PIXEL ERROR: x = {good_avg_x_error:.8f} px, y = {good_avg_y_error:.8f} px", flush=True)

        print("\nTrue pixel centers:", flush=True)
        for i, (x, y) in enumerate(true_centers):
            print(f"  Dot {i+1}: x = {x:.2f}, y = {y:.2f}", flush=True)

        print("\nPredicted pixel centers (good method):", flush=True)
        for i, (x, y) in enumerate(matched_good):
            print(f"  Dot {i+1}: x = {x:.2f}, y = {y:.2f}", flush=True)


        print("\nTrue centers (lat/lon):", flush=True)
        for i, (lat, lon) in enumerate(true_latlons):
            print(f"  Dot {i+1}: Latitude = {lat:.5f}, Longitude = {lon:.5f}", flush=True)

        print("\nPredicted lat/lon (good method):", flush=True)
        for i, (lat, lon) in enumerate(good_pred_latlons):
            print(f"  Dot {i+1}: Latitude = {lat:.5f}, Longitude = {lon:.5f}", flush=True)

        print(f"\nGOOD GEO ERROR (total): {good_geo_error:.2f} meters", flush=True)
        print("\nIndividual GEO errors (good method) in meters:", flush=True)
        for i, err in enumerate(good_geo_errors):
            print(f"  Dot {i+1}: {err:.2f} m", flush=True)

        print(f"\nGOOD AVG LATITUDE ERROR: {good_avg_lat_error:.6f} degrees", flush=True)
        print(f"GOOD AVG LONGITUDE ERROR: {good_avg_lon_error:.6f} degrees", flush=True)
        print(f"\nGOOD AVG DISTANCE ERROR: {good_geo_error:.2f} meters", flush=True)

if __name__ == "__main__":
    with open("dot_center_results.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            start_time = time.time()
            main()
            end_time = time.time()
            print(f"{end_time - start_time:.2f} seconds")

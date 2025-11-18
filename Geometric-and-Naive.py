import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image
from collections import Counter, defaultdict, deque
import cartopy.feature as cfeature
import random
import math
from geopy.distance import geodesic
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
from itertools import product
import sys
from contextlib import redirect_stdout
import time 
from xyzservices import TileProvider
from xyzservices import TileProvider

FILENAME = "Map.png" # Change with Map file name
COORDJSON = "Coord.geojson" # Change with points name
flag=True # Flag = True means PixelMatch Naive Method (can also be used for Raster2Vec)
          # Flag = False means PixelAvg Naive Method
pixel_size = 0.02587884152408056 # Latitude / Pixel
width_px = 2284 # Pixels / Width
height_px = 1424 # Pixels / Height
min_lon = -126.17658145147592563 # Longitude Bound
max_lat = 58.62037301762128294 # Latitude Bound
max_lon = min_lon + pixel_size * width_px
min_lat = max_lat - pixel_size * height_px

DOT_COLOR_RGB = (255, 0, 0) 
COLOR_THRESHOLD = 0 # Tolerance to dot color change (for JPEG)

def latlon_to_pixel(lat, lon, img_width, img_height):
    x = (lon - min_lon) / (max_lon - min_lon) * img_width
    y = (max_lat - lat) / (max_lat - min_lat) * img_height
    return (x, y)
def pixel_to_latlon(x, y, img_width, img_height):
    lon = min_lon + (x / img_width) * (max_lon - min_lon)
    lat = max_lat - (y / img_height) * (max_lat - min_lat)
    return (lat, lon)
def color_distance(c1, c2): # Color distance between two color RGB tuples
    return sum(abs((a) - (b)) for a, b in zip(c1, c2))
def region_area(r, theta): # Used for Circle Geometric Refinement (Can also be applied to Pentagon shape)
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
def region_area_triangle(r, theta): # Used for Triangle Geometric Refinement
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


def geometric_find_dot_centers(image_path, dot_color=DOT_COLOR_RGB):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = img.load()

    global_visited = set()
    dots_info = []
    dot_radii = []

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def first_bfs(start_x, start_y):
        local_visited = set()
        cluster = []
        border_pixels = set()
        pixel_weight = defaultdict(lambda: 1.0)

        queue = deque()
        local_visited.add((start_x, start_y))
        queue.append((start_x, start_y))
        cluster.append((start_x, start_y))

        while queue:
            a, b = queue.popleft()
            for dx, dy in product(range(-1, 2), range(-1, 2)):
                nx, ny = a + dx, b + dy
                if not in_bounds(nx, ny):
                    continue
                if (nx, ny) in local_visited:
                    continue
                local_visited.add((nx, ny))
                if color_distance(pixels[nx, ny], dot_color) <= COLOR_THRESHOLD:
                    queue.append((nx, ny))
                    cluster.append((nx, ny))
                else:
                    cluster.append((nx, ny))
                    border_pixels.add((nx, ny))

        for a, b in border_pixels:
            neighborhood = []
            for dx, dy in product(range(-1, 2), range(-1, 2)):
                na, nb = a + dx, b + dy
                if in_bounds(na, nb) and (na, nb) not in local_visited and (na, nb) not in border_pixels:
                    neighborhood.append(pixels[na, nb])
            if not neighborhood:
                continue
            avg_bg = tuple(sum(p[i] for p in neighborhood) / len(neighborhood) for i in range(3))
            dist_to_bg = color_distance(avg_bg, pixels[a, b])
            dist_to_dot = color_distance(avg_bg, dot_color)
            denom = max(1e-5, dist_to_dot)
            pixel_weight[(a, b)] = min(1.0, dist_to_bg / denom)

        x_sum = y_sum = weight_sum = 0.0
        for x, y in cluster:
            w = pixel_weight[(x, y)]
            x_sum += w * x
            y_sum += w * y
            weight_sum += w

        if weight_sum < 1e-5:
            return None, None, None, None

        cx = x_sum / weight_sum + 0.5
        cy = y_sum / weight_sum + 0.5
        radius = math.sqrt(weight_sum / math.pi)
        return cx, cy, radius, cluster

    def second_bfs(start_x, start_y, approx_x, approx_y, approx_radius, threshold): # Optional - Used for Geometric Refinement (remove weights of bad boundary pixels)
        local_visited = set()
        cluster = []
        border_pixels = set()
        pixel_weight = defaultdict(lambda: 1.0)

        queue = deque()
        local_visited.add((start_x, start_y))
        queue.append((start_x, start_y))
        cluster.append((start_x, start_y))

        while queue:
            a, b = queue.popleft()
            for dx, dy in product(range(-1, 2), range(-1, 2)):
                nx, ny = a + dx, b + dy
                if not in_bounds(nx, ny):
                    continue
                if (nx, ny) in local_visited:
                    continue
                local_visited.add((nx, ny))
                if color_distance(pixels[nx, ny], dot_color) <= COLOR_THRESHOLD:
                    queue.append((nx, ny))
                    cluster.append((nx, ny))
                else:
                    border_pixels.add((nx, ny))
                    cluster.append((nx, ny))

        for a, b in border_pixels:
            dx = abs((a + 0.5) - approx_x)
            dy = abs((b + 0.5) - approx_y)
            dist = math.sqrt(dx ** 2 + dy ** 2)
            theta = 0
            r = 0
            geo_weight = 0
            theta = math.atan2(dy, dx)
            r = approx_radius - dist
            geo_weight = region_area(r, theta) # Change to region_triangle_area if shape is triangle

            neighborhood = []
            for dx2, dy2 in product(range(-1, 2), range(-1, 2)):
                na, nb = a + dx2, b + dy2
                if in_bounds(na, nb) and (na, nb) not in local_visited and (na, nb) not in border_pixels:
                    neighborhood.append(pixels[na, nb])

            if neighborhood:
                avg_bg = tuple(sum(p[i] for p in neighborhood) / len(neighborhood) for i in range(3))
                dist_to_bg = color_distance(avg_bg, pixels[a, b])
                dist_to_dot = color_distance(avg_bg, dot_color)
                denom = max(1e-5, dist_to_dot)
                trad_weight = dist_to_bg / denom
            else:
                trad_weight = geo_weight

            if abs(trad_weight - geo_weight) > threshold:
                pixel_weight[(a, b)] = geo_weight
            else:
                pixel_weight[(a, b)] = trad_weight


        x_sum = y_sum = weight_sum = 0.0
        for x, y in cluster:
            w = pixel_weight[(x, y)]
            x_sum += w * x
            y_sum += w * y
            weight_sum += w

        if weight_sum < 1e-5:
            return None, None
        return x_sum / weight_sum + 0.5, y_sum / weight_sum + 0.5

    for x in range(width):
        for y in range(height):
            if (x, y) in global_visited:
                continue

            if color_distance(pixels[x, y], dot_color) <= COLOR_THRESHOLD:
                cx, cy, radius, cluster_pixels = first_bfs(x, y)
                if cx is None:
                    continue
                dots_info.append((cx, cy, radius, cluster_pixels))
                dot_radii.append(radius)
                global_visited.update(cluster_pixels)
                                                                                                                                                            
    if not dot_radii:
        return []  # no dots found

    average_radius = sum(dot_radii) / len(dot_radii)
    print(f"Average radius across dots: {average_radius}")
    ''' 
    # Uncomment if using Geometric Pixel Filtering
    refined_centers = []
    thresholds = [0.3, 0.15, 0.1, 0.05]
    counter = 0
    for (cx, cy, radius, cluster_pixels) in dots_info:
        counter += 1
        refined_cx, refined_cy = cx, cy
        for threshold in thresholds:
            new_center = second_bfs(int(refined_cx), int(refined_cy), refined_cx, refined_cy, average_radius, threshold)
            if new_center[0] is None:
                break
            refined_cx, refined_cy = new_center
        refined_centers.append((refined_cx, refined_cy))

    return refined_centers
    '''
    return [(cx, cy) for (cx, cy, radius, cluster_pixels) in dots_info]
def naive_find_dot_centers(image_path, dot_color=DOT_COLOR_RGB):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = img.load()

    visited = set()
    predicted_centers = []

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def bfs(x, y):
        cluster = []
        queue = deque()
        visited.add((x, y))
        queue.append((x, y))
        cluster.append((x, y))
        x_color_sum = 0
        y_color_sum = 0
        weight_sum = 0

        while queue:
            a, b = queue.popleft()
            for da, db in [(x, y) for x in range(-1, 2) for y in range(-1, 2)]:
                nx, ny = a + da, b + db
                if in_bounds(nx, ny) and (nx, ny) not in visited:
                    if color_distance(pixels[nx, ny], dot_color) <= COLOR_THRESHOLD:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        cluster.append((nx, ny))
                    else:
                        if flag:
                            cluster.append((nx, ny))

        for i, j in cluster:
            x_color_sum += 1 * i
            y_color_sum += 1 * j
            weight_sum += 1

        return (x_color_sum / weight_sum + 0.5, y_color_sum / weight_sum + 0.5)

    for x in range(width):
        for y in range(height):
            if (x, y) in visited or color_distance(pixels[x, y], dot_color) > COLOR_THRESHOLD:
                continue
            dot_center = bfs(x, y)
            predicted_centers.append(dot_center)
    return predicted_centers


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
            errors.append((abs(true[0] - est[0]), abs(true[1] - est[1])))
            matched.append(tuple(est))
    return errors, matched

def get_true_points(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    
    true_points = []
    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates")
        if coords:
            true_points.append((coords[0], coords[1]))
    return true_points

def main():
    png_path = FILENAME
    true_locations = get_true_points(COORDJSON)
    img = Image.open(png_path)
    img_width, img_height = img.size

    true_centers = [latlon_to_pixel(loc[1], loc[0], img_width, img_height) for loc in true_locations]
    estimated_centers = geometric_find_dot_centers(png_path)
    naive_estimated_centers = naive_find_dot_centers(png_path)

    good_errors, matched_good = match_estimates(true_centers, estimated_centers)
    bad_errors, matched_bad = match_estimates(true_centers, naive_estimated_centers)

    true_latlons = [(loc[1], loc[0]) for loc in true_locations]
    good_pred_latlons = [pixel_to_latlon(x, y, img_width, img_height) for x, y in matched_good]
    bad_pred_latlons = [pixel_to_latlon(x, y, img_width, img_height) for x, y in matched_bad]

    good_geo_errors = [geodesic(true, pred).meters for true, pred in zip(true_latlons, good_pred_latlons)]
    bad_geo_errors = [geodesic(true, pred).meters for true, pred in zip(true_latlons, bad_pred_latlons)]

    def average_coord_error(true_coords, pred_coords):
        lat_errors = [abs(pred[0] - true[0]) for true, pred in zip(true_coords, pred_coords)]
        lon_errors = [abs(pred[1] - true[1]) for true, pred in zip(true_coords, pred_coords)]
        avg_lat_error = sum(lat_errors) / len(lat_errors)
        avg_lon_error = sum(lon_errors) / len(lon_errors)
        return avg_lat_error, avg_lon_error

    def average_pixel_error(true_pixels, pred_pixels):
        x_errors = [abs(pred[0] - true[0]) for true, pred in zip(true_pixels, pred_pixels)]
        y_errors = [abs(pred[1] - true[1]) for true, pred in zip(true_pixels, pred_pixels)]
        avg_x_error = sum(x_errors) / len(x_errors)
        avg_y_error = sum(y_errors) / len(y_errors)
        return avg_x_error, avg_y_error

    def average(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def average_pixel_displacement_components(true_pixels, pred_pixels):
        dxs = [pred[0] - true[0] for true, pred in zip(true_pixels, pred_pixels)]
        dys = [pred[1] - true[1] for true, pred in zip(true_pixels, pred_pixels)]
        avg_dx = sum(dxs) / len(dxs) if dxs else 0.0
        avg_dy = sum(dys) / len(dys) if dys else 0.0
        return avg_dx, avg_dy

    good_avg_lat_error, good_avg_lon_error = average_coord_error(true_latlons, good_pred_latlons)
    bad_avg_lat_error, bad_avg_lon_error = average_coord_error(true_latlons, bad_pred_latlons)

    good_avg_x_error, good_avg_y_error = average_pixel_error(true_centers, matched_good)
    bad_avg_x_error, bad_avg_y_error = average_pixel_error(true_centers, matched_bad)

    good_avg_dx, good_avg_dy = average_pixel_displacement_components(true_centers, matched_good)
    bad_avg_dx, bad_avg_dy = average_pixel_displacement_components(true_centers, matched_bad)

    good_geo_error = average(good_geo_errors)
    bad_geo_error = average(bad_geo_errors)

    print("\nTrue pixel centers:", flush=True)
    for i, (x, y) in enumerate(true_centers):
        print(f"  Dot {i+1}: x = {x:.2f}, y = {y:.2f}", flush=True)

    print("\nPredicted pixel centers (good method):", flush=True)
    for i, (x, y) in enumerate(matched_good):
        print(f"  Dot {i+1}: x = {x:.2f}, y = {y:.2f}", flush=True)

    print("\nPredicted pixel centers (naive method):", flush=True)
    for i, (x, y) in enumerate(matched_bad):
        print(f"  Dot {i+1}: x = {x:.2f}, y = {y:.2f}", flush=True)

    print(f"\nGOOD AVG PIXEL ERROR: x = {good_avg_x_error:.3f} px, y = {good_avg_y_error:.3f} px", flush=True)
    print(f"NAIVE AVG PIXEL ERROR: x = {bad_avg_x_error:.3f} px, y = {bad_avg_y_error:.3f} px", flush=True)
    print(f"GOOD AVG PIXEL DISPLACEMENT: dx = {good_avg_dx:.6f} px, dy = {good_avg_dy:.6f} px", flush=True)
    print(f"NAIVE AVG PIXEL DISPLACEMENT: dx = {bad_avg_dx:.3f} px, dy = {bad_avg_dy:.3f} px", flush=True)

    print("\nTrue centers (lat/lon):", flush=True)
    for i, (lat, lon) in enumerate(true_latlons):
        print(f"  Dot {i+1}: Latitude = {lat:.5f}, Longitude = {lon:.5f}", flush=True)

    print("\nPredicted lat/lon (good method):", flush=True)
    for i, (lat, lon) in enumerate(good_pred_latlons):
        print(f"  Dot {i+1}: Latitude = {lat:.5f}, Longitude = {lon:.5f}", flush=True)

    print("\nPredicted lat/lon (naive method):", flush=True)
    for i, (lat, lon) in enumerate(bad_pred_latlons):
        print(f"  Dot {i+1}: Latitude = {lat:.5f}, Longitude = {lon:.5f}", flush=True)

    print(f"\nGOOD GEO ERROR (total): {good_geo_error:.2f} meters", flush=True)
    print(f"NAIVE GEO ERROR (total): {bad_geo_error:.2f} meters", flush=True)

    print("\nIndividual GEO errors (good method) in meters:", flush=True)
    for i, err in enumerate(good_geo_errors):
        print(f"  Dot {i+1}: {err:.2f} m", flush=True)

    print("\nIndividual GEO errors (naive method) in meters:", flush=True)
    for i, err in enumerate(bad_geo_errors):
        print(f"  Dot {i+1}: {err:.2f} m", flush=True)

    print(f"GOOD AVG PIXEL ERROR: {good_avg_x_error:.6} {good_avg_y_error:.6}")
    print(f"BAD AVG PIXEL ERROR: {bad_avg_x_error:.6} {bad_avg_y_error:.6}")

    print(f"\nGOOD AVG LATITUDE ERROR: {good_avg_lat_error:.6f} degrees", flush=True)
    print(f"GOOD AVG LONGITUDE ERROR: {good_avg_lon_error:.6f} degrees", flush=True)
    print(f"\nBAD AVG LATITUDE ERROR: {bad_avg_lat_error:.6f} degrees", flush=True)
    print(f"BAD AVG LONGITUDE ERROR: {bad_avg_lon_error:.6f} degrees", flush=True)

    print(f"\nGOOD AVG DISTANCE ERROR: {good_geo_error:.2f} meters", flush=True)
    print(f"BAD AVG DISTANCE ERROR: {bad_geo_error:.2f} meters", flush=True)



if __name__ == "__main__":
    with open("dot_center_results.txt", "w") as f:
        with redirect_stdout(f):
            start_time = time.time()
            main()
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total runtime: {total_time:.2f} seconds")

import json

'''
Quantizes every coordinate in input file to n decimals
'''
input_file  = "map.geojson"
output_file = "output.geojson"
n = 2                             

def round_coords(coords, n):
    if isinstance(coords[0], (float, int)):
        return [round(coords[0], n), round(coords[1], n)]
    else:
        return [round_coords(c, n) for c in coords]

def main():
    with open(input_file, "r") as f:
        data = json.load(f)

    for feature in data["features"]:
        geom = feature["geometry"]
        geom["coordinates"] = round_coords(geom["coordinates"], n)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done! Rounded coords written to: {output_file}")

if __name__ == "__main__":
    main()

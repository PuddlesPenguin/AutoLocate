from PIL import Image
import random

'''
This script perturbs every pixel in the image by a color distance of 10
'''

DOT_COLOR = (255, 0, 0)
input_path = "map.png"
output_path = "output.png"

def rand_delta():
    d = random.randint(1, 10)
    if random.random() < 0.5:
        return -d
    return d

def perturb_non_red(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    pixels = img.load()

    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            if (r, g, b) == DOT_COLOR:
                continue

            dr = rand_delta()
            dg = rand_delta()
            db = rand_delta()

            new_r = max(0, min(255, r + dr))
            new_g = max(0, min(255, g + dg))
            new_b = max(0, min(255, b + db))

            if (new_r, new_g, new_b) == DOT_COLOR:
                if new_g < 255:
                    new_g += 1
                else:
                    new_g -= 1

            pixels[x, y] = (new_r, new_g, new_b)

    img.save(output_path)
    print("Saved:", output_path)
perturb_non_red(input_path, output_path)

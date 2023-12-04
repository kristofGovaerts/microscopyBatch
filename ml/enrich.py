"""
This code enriches data in a folder by adding random flips and brightness adjustments. Should be run exactly once!
Enriched files have the extension '_E'.
"""

import glob
import os
import numpy as np
import cv2

LOCATION = r'C:\Users\govaerts.kristof\programming\cuisson\0'
ENRICH_FACTOR = 3  # an integer value. a fraction of the files you want to enrich. 1 results in a dataset that is twice as large (ie 100% enrichment). A value of 3 means ~33% extra images.


def adjust_brightness(image, scale_factor):
    image = image.astype(np.float32)
    scaled = image * scale_factor

    return scaled.astype(np.uint8)


def flip_image(image, flip_row=False, flip_col=False):
    if flip_row:
        image = np.flip(image, 0)

    if flip_col:
        image = np.flip(image, 1)

    return image


def random_flips(images):
    flip_row = (np.random.randint(2) == 1)
    flip_col = (np.random.randint(2) == 1)

    results = []
    for image in images:
        result = flip_image(image, flip_row, flip_col)
        results.append(result)

    return results


if __name__ == '__main__':
    os.chdir(LOCATION)
    fl = glob.glob('*.png')

    for f in fl:
        if np.random.randint(ENRICH_FACTOR) == 1:
            img = cv2.imread(f)
            img = random_flips([img])[0]
            brightness = np.random.uniform(0.7, 0.99)
            image = adjust_brightness(img, brightness)
            cv2.imwrite(f.replace('.png', '_E.png'), image)

"""
This script will merge the classes of the initial dataset, as the number of classes is too high.
The scope is to reach 5 classes instead of 23:
    -> Ground: 1 (paved: 1, dirt: 2)
    -> Human: 2 (original: 15)
    -> Car: 3 (original: 17)
    -> Vegetation: 4 (tree: 19, gras: 3, other:8, dirt:2)
    -> Water: 5 (original: 5)
    -> Roof: 6 (original: 9)
    -> Other: 7 (excluded)
"""
import os
import cv2
import sys
import numpy as np


def process_images(input_folder, output_folder):

    def process(img):
        new_image = np.zeros_like(img)
        # Paved floor and dirt
        new_image[img == 1] = 1
        new_image[img == 2] = 1
        # Human
        new_image[img == 15] = 2
        # Car
        new_image[img == 17] = 3
        # Vegetation
        new_image[img == 19] = 4
        new_image[img == 3] = 4
        new_image[img == 8] = 4
        # Water
        new_image[img == 5] = 5
        # Roof
        new_image[img == 9] = 6
        # Other classes are excluded
        new_image[new_image == 0] = 7
        return new_image

    files = os.listdir(input_folder)
    for file in files:
        image = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_GRAYSCALE)
        processed = process(image)
        cv2.imwrite(os.path.join(output_folder, file), processed)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('Usage: python merge_classes.py <input_folder> <output_folder>')

    process_images(sys.argv[1], sys.argv[2])

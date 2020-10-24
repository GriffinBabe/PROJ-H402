import numpy as np
import cv2
import os

from preprocessing import load_paths


def select_complex_labels(label_dir, label_output_dir=None, image_dir=None, image_output_dir=None, min_types=2):
    """
    Selects from a folder the most complex labeled image.
    This is useful to reduce the training of the model on
    road only parts of the data.
    :param: image_dir, the directory of the images.
    :return: A list with the selected image paths.
    """
    processed_images = 0
    values = {}
    for im_p in load_paths(label_dir):
        path = os.path.join(label_dir, im_p)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        histogram = np.histogram(im, 256)[0]
        value_types = 0
        for val in histogram:
            if val != 0:
                value_types += 1
        if value_types >= min_types:
            values[path] = value_types
        processed_images += 1
        print("Processed image {}".format(im_p))
    print("{} images have been processed.".format(processed_images))
    if label_output_dir is not None and image_dir is not None and image_output_dir is not None:
        for im_p in values.keys():
            print("Saving image: {}".format(im_p))
            with open(os.path.join(label_dir, im_p), 'rb') as src, open(os.path.join(label_output_dir, im_p), 'wb') as dst:
                dst.write(src.read())
            with open(os.path.join(image_dir, im_p), 'rb') as src, open(os.path.join(image_output_dir, im_p), 'wb') as dst:
                dst.write(src.read())
    return values

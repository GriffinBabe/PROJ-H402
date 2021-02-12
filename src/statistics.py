import numpy as np
import cv2
import os

from src.preprocessing import load_paths


def save_selected_files(values, label_dir, image_dir, label_output_dir, image_output_dir):
    """
    Copies all the files in the values dicrionary from the input directory to the output directory.
    Does that for the images and the respective labels.
    :param values:
    :param label_dir:
    :param image_dir:
    :param label_output_dir:
    :param image_output_dir:
    """
    for im_p in values.keys():
        print("Saving image: {}".format(im_p))
        with open(os.path.join(label_dir, im_p), 'rb') as src, open(os.path.join(label_output_dir, im_p), 'wb') as dst:
            dst.write(src.read())
        with open(os.path.join(image_dir, im_p.replace('png', 'jpg')), 'rb') as src, open(os.path.join(image_output_dir, im_p), 'wb') as dst:
            dst.write(src.read())


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
        save_selected_files(values, label_dir, image_dir, label_output_dir, image_output_dir)
    return values


def select_complex_labels_road_density(label_dir, label_output_dir=None, image_dir=None, image_output_dir=None, max_perc=0.70):
    """
    Selects from a folder the image that have the least road surface on the label, based on a threshold percentage.
    :param: label_dir, The directory to the image labels
    :param: label_output_dir, Output directory of selected labels
    :param: image_dir, The directory of the related images
    :param: image_output_dir, Output directory of related images
    :param: max_perc, Max percentage of surface road for an image to be accepted.
    :return: A list with the selected image paths.
    """
    processed_images = 0
    values = {}
    for im_p in load_paths(label_dir):
        path = os.path.join(label_dir, im_p)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        histogram = np.histogram(im, 256)[0]
        road_percentage = histogram[0] / (im.shape[0] * im.shape[1])
        print('Road perc: {}%'.format(road_percentage * 100.0))
        if road_percentage <= max_perc:
            print('Selected')
            values[im_p] = road_percentage
        else:
            print('Not selected')
        processed_images += 1
        print('Processed image {}'.format(im_p))
        print('{} images have been processed.'.format(processed_images))
    if label_output_dir is not None and image_dir is not None and image_output_dir is not None:
        save_selected_files(values, label_dir, image_dir, label_output_dir, image_output_dir)
    return values

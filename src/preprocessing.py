import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt


def load_paths(directory_path, exclude=None):
    """
    Loads all the paths in a directory and returns a list of them.
    :param: directory_path, the path to a directory.
    :param: exclude, optional, list of substring file names to exclude
    :returns: an array of all paths to the files in that directory.
    """
    paths = []
    for f in os.listdir(directory_path):
        if not os.path.isfile(os.path.join(directory_path, f)):
            continue
        if exclude is not None:
            for subs in exclude:
                if subs in f:
                    continue
        paths.append(f)
    return paths


def rescale_images(directory_path, output_path, new_size, image_suffix='_resized'):
    image_paths = load_paths(directory_path, image_suffix)
    images_count = len(image_paths)
    print("Found {} images_splitted in directory: {}".format(images_count, output_path))
    processed_count = 0
    for im_p in image_paths:
        print("Processing image {}/{}.".format(processed_count + 1, images_count))
        im = cv2.imread(os.path.join(directory_path, im_p), cv2.IMREAD_COLOR)
        im_resized = cv2.resize(im, new_size)
        cv2.imwrite(os.path.join(output_path, im_p), im_resized)
        processed_count += 1


def flip_images(directory_path, output_path, image_suffix='_flipped', horizontal=True):
    image_paths = load_paths(directory_path, image_suffix)
    images_count = len(image_paths)
    print('Found {} images_splitted in directory: {}'.format(images_count, output_path))
    processed_count = 0
    for im_p in image_paths:
        print("Processing image {}/{}.".format(processed_count + 1, images_count))
        title_regex = r'(.*)\.([A-Za-z0-9]+)$'
        file_name = re.search(title_regex, im_p).group(1)
        file_extension = re.search(title_regex, im_p).group(2)
        file_new_name = file_name+image_suffix+'.'+file_extension
        im = cv2.imread(os.path.join(directory_path, im_p), cv2.IMREAD_COLOR)
        if horizontal:
            im_flipped = cv2.flip(im, 1)
        else:
            im_flipped = cv2.flip(im, 0)
        cv2.imwrite(os.path.join(output_path, file_new_name), im_flipped)
        processed_count += 1


def split_images(directory_path, output_path, width_div, height_div):
    image_paths = load_paths(directory_path, None)
    im = cv2.imread(os.path.join(directory_path, image_paths[0]))
    im_size = im.shape
    sub_im_h = int(im_size[0] / height_div)
    sub_im_w = int(im_size[1] / width_div)
    processed_count = 0
    total_images = len(image_paths) * height_div * width_div
    for im_p in image_paths:
        title_regex = r'(.*)\.([A-Za-z0-9]+)$'
        file_name = re.search(title_regex, im_p).group(1)
        file_extension = re.search(title_regex, im_p).group(2)
        im = cv2.imread(os.path.join(directory_path, im_p), cv2.IMREAD_COLOR)
        for x in range(height_div):
            for y in range(width_div):
                print(im.shape)
                print('Processing image {}/{}'.format(processed_count, total_images))
                sub_image = im[x*sub_im_h:(x + 1)*sub_im_h, y*sub_im_w:(y + 1)*sub_im_w, :]
                file_new_name = '{}_s{}_{}.{}'.format(file_name, x, y, file_extension)
                cv2.imwrite(os.path.join(output_path, file_new_name), sub_image)
                processed_count += 1


def preprocess_images_and_labels():
    rescale_images('../dataset/original_images',
                   'dataset/processed_images/images_rescaled',
                   (5632, 3584))
    flip_images('dataset/processed_images/images_splitted',
                'dataset/processed_images/images_splitted',
                image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images',
                'dataset/processed_images',
                image_suffix='_vflipped', horizontal=False)
    rescale_images('../dataset/label_images_semantic',
                   'dataset/processed_images/label_rescaled',
                   (5632, 3584))
    flip_images('dataset/processed_images/label_splitted',
                'dataset/processed_images/label_splitted',
                image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images/label_splitted',
                'dataset/processed_images/label_splitted',
                image_suffix='_vflipped', horizontal=False)

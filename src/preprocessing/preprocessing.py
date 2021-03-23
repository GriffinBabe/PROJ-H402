import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt
from skimage.transform import resize


def process_segmentation_output(output, output_shape, n_classes):
    """
    Converts the segmentation output into a visible image
    :return:
    """

    # Color linspace for segmentation output presentation
    t = np.linspace(-510, 510, n_classes)
    color_array = np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)

    output_list = []

    for out in output:
        out = out.reshape((output_shape[0], output_shape[1], n_classes))
        # Merge each channel into an image
        out_processed = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
        out_processed[:, :] = color_array[np.argmax(out, axis=2)]
        output_list.append(out_processed)

    return output_list


def preprocess_segmentation(image, input_shape, n_classes):
    """
    Processes a segmentation image into multiple channels input map for the UNET
    :param image: a uchar8 image with n_classes
    :return:
    """
    image = resize(image, input_shape)
    if n_classes > 256 or isinstance(image[0, 0].dtype, np.uint8):
        raise Exception('Image must be an uint8 image and channels cannot be more than 256')
    seg = np.zeros((image.shape[0], image.shape[1], n_classes), dtype=np.float)
    for i in range(n_classes):
        seg[image[:, :] == i, i] = 1.0
    return seg


def preprocess_rgb_image(image, input_shape):
    """
    Processes an input RGB image for the unet
    :param image:
    :return:
    """
    image = resize(image, input_shape)
    if isinstance(image[0, 0].dtype, np.uint8):
        return image.astype(np.float) / 255.0
    else:
        return image


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
    rescale_images('../../dataset/original_images',
                   'dataset/processed_images/images_rescaled',
                   (5632, 3584))
    flip_images('dataset/processed_images/images_splitted',
                'dataset/processed_images/images_splitted',
                image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images',
                'dataset/processed_images',
                image_suffix='_vflipped', horizontal=False)
    rescale_images('../../dataset/label_images_semantic',
                   'dataset/processed_images/label_rescaled',
                   (5632, 3584))
    flip_images('dataset/processed_images/label_splitted',
                'dataset/processed_images/label_splitted',
                image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images/label_splitted',
                'dataset/processed_images/label_splitted',
                image_suffix='_vflipped', horizontal=False)

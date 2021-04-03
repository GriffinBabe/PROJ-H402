"""
Data augmentation is a necessary step on small datasets such as this one.
This file contains methods to add new data by applying artificial transformations to the image.
"""

import sys
import os
import cv2
import numpy as np
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import rotate
import random


def augment_dataset(src_image_dir, src_label_dir, dst_image_dir, dst_label_dir):
    """
    Augment the images by applying horizontal flip, vertical flip and random rotations.
    Input and label images are transformed the same way.

    :param src_image_dir: source directory for input images
    :param src_label_dir: source directory for labels
    :param dst_image_dir: destination directory for augmented input images
    :param dst_label_dir: destination directory for augmented label
    :return:
    """

    input_image_files = os.listdir(src_image_dir)
    label_image_files = os.listdir(src_label_dir)
    cnt = 0
    total = len(input_image_files)
    for img_file, label_file in zip(input_image_files, label_image_files):
        print('Processing image: {}/{}'.format(cnt + 1, total))
        img_path = os.path.join(src_image_dir, img_file)
        label_path = os.path.join(src_label_dir, label_file)

        # rotation = int(random.uniform(-20, 20))

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image_v_flipped = np.flipud(image)
        image_h_flipped = np.fliplr(image)
        # image_rotated = rotate(image, rotation)

        label_v_flipped = np.flipud(label)
        label_h_flipped = np.fliplr(label)
        # label_rotated = rotate(label, rotation)

        img_file = img_file.split('.')
        cv2.imwrite(os.path.join(dst_image_dir, img_file[0]+'_0.'+img_file[1]), image)
        cv2.imwrite(os.path.join(dst_image_dir, img_file[0]+'_1.'+img_file[1]), image_v_flipped)
        cv2.imwrite(os.path.join(dst_image_dir, img_file[0]+'_2.'+img_file[1]), image_h_flipped)
        # cv2.imwrite(os.path.join(dst_image_dir, img_file[0]+'_3.'+img_file[1]), image_rotated)

        label_file = label_file.split('.')
        cv2.imwrite(os.path.join(dst_label_dir, label_file[0]+'_0.'+label_file[1]), label)
        cv2.imwrite(os.path.join(dst_label_dir, label_file[0]+'_1.'+label_file[1]), label_v_flipped)
        cv2.imwrite(os.path.join(dst_label_dir, label_file[0]+'_2.'+label_file[1]), label_h_flipped)
        # cv2.imwrite(os.path.join(dst_label_dir, label_file[0]+'_3.'+label_file[1]), label_rotated)

        cnt += 1


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise Exception('Usage: python augmentation.py <src_img_dir> <src_label_dir> <dst_img_dir> <dst_label_dir>')

    augment_dataset(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

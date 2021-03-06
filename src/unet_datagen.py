"""
Data generators for U-Net. Data generators are iterators that let the model train over the whole dataset without needing
to load every element at once. This is useful when not enough memory is available. Preprocessing and data augmentation
tasks are also possible through data generators.

ImageDataGenerator class already exists for Keras, but does not support a generic number of channels. In the case
of drone dataset, 23 different classes are needed.
"""
from keras.utils import Sequence
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import shuffle
import cv2
import os


class UnetDataGenerator(Sequence):

    def __init__(self, input_directory, label_directory, size, label_channels, batch_size, shuffle=True):
        self._size = size
        self._label_channels = label_channels
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._image_dir = os.path.join(input_directory, 'img/')
        self._label_dir = os.path.join(label_directory, 'img/')
        self._image_paths = os.listdir(self._image_dir)
        self._label_paths = os.listdir(self._label_dir)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self._temp_label_paths) / self._batch_size))

    def __data_generation(self, seg_image):
        # Resizing will already be done by the Generator
        out = np.zeros((seg_image.shape[0], seg_image.shape[1], self._label_channels), dtype=np.uint8)
        for i in range(self._label_channels):
            mask = (seg_image[:, :] == i)
            out[mask, i] = 1
        return out

    def __getitem__(self, item):
        self._temp_image_paths, image_paths = self._temp_image_paths[self._batch_size:], self._temp_image_paths[:self._batch_size]
        self._temp_label_paths, label_paths = self._temp_label_paths[self._batch_size:], self._temp_label_paths[:self._batch_size]
        image_batch = []
        label_batch = []
        for idx in range(len(image_paths)):
            image = imread(os.path.join(self._image_dir, image_paths[idx]))
            image = resize(image, self._size)
            image_batch.append(image)

            label = cv2.imread(os.path.join(self._label_dir, label_paths[idx]), cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (self._size[1], self._size[0]))
            label = self.__data_generation(label)  # Transforms single channel grayscale image into multiple channel image
            label_batch.append(label)
        return np.array(image_batch, dtype=np.uint8), np.array(label_batch, dtype=np.uint8)

    def on_epoch_end(self):
        # Called at the end of each epoch
        self._temp_image_paths = list(self._image_paths)
        self._temp_label_paths = list(self._label_paths)
        if self._shuffle:
            self._temp_image_paths, self._temp_label_paths = shuffle(self._temp_image_paths, self._temp_label_paths)

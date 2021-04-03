"""
Data generators for U-Net. Data generators are iterators that let the model train over the whole dataset without needing
to load every element at once. This is useful when not enough memory is available. Preprocessing and data augmentation
tasks are also possible through data generators.

ImageDataGenerator class already exists for Keras, but does not support a generic number of channels. In the case
of drone dataset, 23 different classes are needed.
"""
from keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
        image_paths = self._temp_image_paths[self._batch_size*item:self._batch_size*(item + 1)]
        label_paths = self._temp_label_paths[self._batch_size*item:self._batch_size*(item + 1)]
        image_batch = []
        label_batch = []
        for idx in range(len(image_paths)):
            image = imread(os.path.join(self._image_dir, image_paths[idx]))
            image = resize(image, self._size)
            image_batch.append(image)

            label = cv2.imread(os.path.join(self._label_dir, label_paths[idx]), cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (int(self._size[1] / 2), int(self._size[0] / 2)))
            label = label.reshape(int(self._size[1] / 2) * int(self._size[0] / 2), -1)
            # label = self.__data_generation(label)  # Transforms single channel grayscale image into multiple channel image
            label_batch.append(label)
        return np.array(image_batch, dtype=np.uint8), np.array(label_batch, dtype=np.uint8)

    def on_epoch_end(self):
        # Called at the end of each epoch
        self._temp_image_paths = list(self._image_paths)
        self._temp_label_paths = list(self._label_paths)
        if self._shuffle:
            self._temp_image_paths, self._temp_label_paths = shuffle(self._temp_image_paths, self._temp_label_paths)


def get_default_datagen(image_dir, label_dir, input_size, label_size, batch_size=1):

    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    image_generator = train_datagen.flow_from_directory(image_dir,
                                                        target_size=(input_size[0], input_size[1]),
                                                        batch_size=batch_size,
                                                        seed=42,
                                                        class_mode=None)  # Doesn't expect a specific folder layout

    label_generator = train_datagen.flow_from_directory(label_dir,
                                                        target_size=(label_size[0] * label_size[1], 1),
                                                        batch_size=batch_size,
                                                        seed=42,
                                                        color_mode='grayscale',
                                                        class_mode='sparse')

    train_generator = zip(image_generator, label_generator)  # Merge both generators for fit_generator call

    return train_generator


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
from itertools import cycle
import cv2
import os


class UnetDataGenerator(Sequence):
    """
    Inspired on this thread: https://github.com/keras-team/keras/issues/12120
    """

    def __init__(self, features, targets, image_size, label_size, classes, batch_size, repeats=1, shuffle=True):
        self._features = features
        self._targets = targets
        self._image_size = image_size
        self._label_size = label_size
        self._classes = classes
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._repeats = repeats

        self._ids = list(self._list_files())

        self._indexes = np.arange(len(self._ids))
        if self._shuffle:
            np.random.shuffle(self._indexes)
        self._indexes = np.repeat(self._indexes, self._repeats)

        self._augmenter = ImageDataGenerator(rotation_range=30,
                                             zoom_range=0.2,
                                             width_shift_range=0.01,
                                             height_shift_range=0.01,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             samplewise_std_normalization=True)

    def _list_files(self):
        feature_files = os.listdir(self._features)
        target_files = os.listdir(self._targets)
        return zip(feature_files, target_files)

    def _encode_one_hot(self, label):
        x = np.zeros((self._label_size[0] * self._label_size[1], self._classes))
        for c in range(1, self._classes + 1):
            a = label == c
            x[(label == c).squeeze(), c - 1] = 1
        return x

    def __len__(self):
        return int(np.floor(len(self._ids) / self._batch_size)) * self._repeats

    def __getitem__(self, item):
        indexes = self._indexes[item * self._batch_size: (item + 1) * self._batch_size]
        ids_temp = [self._ids[idx] for idx in indexes]

        X = [cv2.resize(
                cv2.imread(os.path.join(self._features, ids_temp[k][0]), cv2.IMREAD_COLOR), self._image_size, cv2.INTER_NEAREST).astype(np.float64) / 255.0
             for k in range(len(ids_temp))]
        y = [cv2.resize(
                cv2.imread(os.path.join(self._targets, ids_temp[k][1]), cv2.IMREAD_COLOR), self._image_size, cv2.INTER_NEAREST)
             for k in range(len(ids_temp))]

        params = self._augmenter.get_random_transform(self._image_size)

        X = np.array([self._augmenter.apply_transform(self._augmenter.standardize(x), params) for x in X])
        y = np.array([self._encode_one_hot(cv2.resize(self._augmenter.apply_transform(_y, params)[:, :, 0], self._label_size, cv2.INTER_NEAREST).reshape(-1, 1)) for _y in y])

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        self._indexes = np.arange(len(self._ids))
        if self._shuffle:
            np.random.shuffle(self._indexes)
        self._indexes = np.repeat(self._indexes, self._repeats)


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


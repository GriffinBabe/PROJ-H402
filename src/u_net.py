from src.preprocessing import preprocess_segmentation, preprocess_rgb_image, process_segmentation_output
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.global_var import *
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import os

# Implemented from the paper:
# U-Net: Convolutional Networks for Biomedical
# Image Segmentation
# Olaf Ronneberger, Philipp Fischer, and Thomas Brox
# Computer Science Department and BIOSS Centre for Biological Signalling Studies,
# University of Freiburg, Germany
# https://arxiv.org/pdf/1505.04597.pdf


INPUT_WIDTH = 608
INPUT_HEIGHT = 416
N_CLASSES = 23
N_EPOCHS = 2


def u_net(input_height, input_width, n_classes, pretrained_weights=None):
    """
    Initialises a U-NET with Keras layers.
    :param input_height: the input image height
    :param input_width: the input image width
    :param n_classes: the number of classes for semantic segmentation
    :param pretrained_weights: by default None, if specified loads the weights from a file.
    :return: a Keras U-NET model.
    """

    input = keras.Input(shape=(input_height, input_width, 3))

    # Going down in resolution, increasing feature channels

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(input)
    x_1 = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    x = layers.MaxPool2D(2)(x_1)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x_2 = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = layers.MaxPool2D(2)(x_2)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x_3 = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.MaxPool2D(2)(x_3)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x_4 = layers.Conv2D(512, 3, activation='relu', padding='same')(x)

    x = layers.MaxPool2D(2)(x_4)
    x = layers.Conv2D(1024, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(1024, 3, activation='relu', padding='same')(x)

    # Going un in resolution, decreasing feature channels

    x = layers.UpSampling2D((2, 2))(x)
    concat_1 = layers.concatenate((x_4, x), axis=3)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(concat_1)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    concat_2 = layers.concatenate((x_3, x), axis=3)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(concat_2)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    concat_3 = layers.concatenate((x_2, x), axis=3)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(concat_3)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    concat_4 = layers.concatenate((x_1, x), axis=3)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(concat_4)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    output = layers.Conv2D(n_classes, 1, activation='sigmoid')(x)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def get_image_generators(image_dir, label_dir):
    """
    Will create generators to preprocess images and feed them into the model while training.
    The generator are helpful as we don't have to fully load the dataset into memory.
    :return: A generator that will provided processed data for the model.
    """

    def segmentation_preprocessing(seg_image):
        # Resizing will already be done by the Generator
        out = np.zeros((seg_image.shape[0], seg_image.shape[1], N_CLASSES), dtype=np.uint8)
        for i in range(N_CLASSES):
            out[seg_image[:, :] == i, i] = 1.0
        return out

    image_datagen = ImageDataGenerator(
        rescale=1.0/255.0
    )
    label_datagen = ImageDataGenerator(preprocessing_function=segmentation_preprocessing)

    image_generator = image_datagen.flow_from_directory(image_dir,
                                                        target_size=(INPUT_HEIGHT, INPUT_WIDTH),
                                                        batch_size=32,
                                                        class_mode=None)  # Doesn't expect a specific folder layout

    label_generator = label_datagen.flow_from_directory(label_dir,
                                                        target_size=(INPUT_HEIGHT, INPUT_WIDTH),
                                                        batch_size=32,
                                                        class_mode=None)

    train_generator = zip(image_generator, label_generator)  # Merge both generators for fit_generator call

    return train_generator


def train_unet(input_dir, label_dir, epochs):
    model = u_net(INPUT_HEIGHT, INPUT_WIDTH, N_CLASSES)
    generators = get_image_generators(input_dir, label_dir)
    model.fit(generator=generators, epochs=epochs)
    return model


if __name__ == '__main__':
    trained_model = train_unet(ORIGINAL_IMAGES_DIRECTORY, LABELED_IMAGES_DIRECTORY, N_EPOCHS)

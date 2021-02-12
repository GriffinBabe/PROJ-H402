from src.train_vgg import input_width, input_height, n_classes
from src.preprocessing import preprocess_segmentation, preprocess_rgb_image, process_segmentation_output
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


def train_unet(input_dir, label_dir):
    model = u_net(input_height, input_width, n_classes)
    image_1_path = os.path.join(input_dir, '000.jpg')
    seg_1_path = os.path.join(label_dir, '000.png')

    image_1 = preprocess_rgb_image(imread(image_1_path), (input_height, input_width))
    seg_1 = preprocess_segmentation(imread(seg_1_path, as_gray=True), (input_height, input_width), n_classes)

    image_1 = np.expand_dims(image_1, axis=0)
    seg_1 = np.expand_dims(seg_1, axis=0)

    model.fit(x=image_1, y=seg_1, batch_size=1, epochs=2, verbose=True)

    # Predict with the same single input image used for training
    seg_output = model.predict(image_1)
    seg_output = process_segmentation_output(seg_output, (input_height, input_width), n_classes)

    plt.imshow(seg_output[0])
    plt.show()


train_unet(ORIGINAL_IMAGES_DIRECTORY, LABELED_IMAGES_DIRECTORY)

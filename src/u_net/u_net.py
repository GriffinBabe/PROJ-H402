from tensorflow.keras.callbacks.experimental import BackupAndRestore
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from src.u_net.unet_datagen import get_default_datagen
from src.global_var import *
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
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
N_EPOCHS = 20
BATCH_SIZE = 1


def learning_rate(epoch):
    pass


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

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)

    x_1 = layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_last')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(x_1)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)

    x_2 = layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_last')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(x_2)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)

    x_3 = layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_last')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_last')(x_3)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)

    x_4 = layers.MaxPool2D((2, 2), strides=(2, 2), data_format='channels_last')(x)
    x = layers.ZeroPadding2D((1, 1), data_format='channels_last')(x_4)
    x = layers.Conv2D(512, (3, 3), padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    # Going un in resolution, decreasing feature channels

    x = layers.UpSampling2D((2, 2), data_format='channels_last')(x)
    x = layers.concatenate((x, x_3), axis=3)
    x = layers.ZeroPadding2D((1,  1), data_format='channels_last')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2), data_format='channels_last')(x)
    x = layers.concatenate((x, x_2), axis=3)
    x = layers.ZeroPadding2D((1,  1), data_format='channels_last')(x)
    x = layers.Conv2D(128, (3, 3), padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2), data_format='channels_last')(x)
    x = layers.concatenate((x, x_1), axis=3)
    x = layers.ZeroPadding2D((1,  1), data_format='channels_last')(x)
    x = layers.Conv2D(64, (3, 3), padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(n_classes, (1, 1), data_format='channels_last')(x)
    x = layers.Reshape((int(input_height / 2) * int(input_width / 2), -1))(x)
    output = layers.Activation('softmax')(x)

    model = keras.Model(inputs=[input], outputs=[output])
    model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def train_unet(input_dir, label_dir, epochs, save_path=None):
    model = u_net(INPUT_HEIGHT, INPUT_WIDTH, N_CLASSES)
    plot_model(model, to_file='out/u_net_plot.png', show_shapes=True, show_layer_names=True)
    # generators = UnetDataGenerator(input_dir, label_dir,
    #                                (INPUT_HEIGHT, INPUT_WIDTH), N_CLASSES, BATCH_SIZE, shuffle=True)
    generators = get_default_datagen(input_dir, label_dir, (INPUT_WIDTH, INPUT_HEIGHT),
                                     (int(INPUT_WIDTH/2), int(INPUT_HEIGHT/2)))
    checkpoint_path = os.path.join(MODELS_DIRECTORY, UNET_CHECKPOINT_PATH)
    model_directory = os.path.join(MODELS_DIRECTORY, UNET_20_EPOCHS)

    backup_callback = BackupAndRestore(backup_dir=checkpoint_path)
    # In case of a plateau, reducing the learning rate by 2 can help
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=2, min_lr=0.0001)
    model.fit(x=generators, epochs=epochs, steps_per_epoch=400, callbacks=[backup_callback, reduce_lr])
    # Saves the weights
    if save_path is not None:
        model.save(save_path)
    return model


if __name__ == '__main__':
    model_save_path = os.path.join(MODELS_DIRECTORY, UNET_20_EPOCHS)
    trained_model = train_unet(ORIGINAL_IMAGES_DIRECTORY, LABELED_IMAGES_DIRECTORY, N_EPOCHS, save_path=model_save_path)

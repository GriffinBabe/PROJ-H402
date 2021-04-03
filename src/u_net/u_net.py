from keras.utils.vis_utils import plot_model
from src.u_net.unet_datagen import get_default_datagen
from src.global_var import *
from src.utils.custom_callbacks import SaveWeightCallback, PrintLR
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os
from keras_segmentation.models.vgg16 import get_vgg_encoder

# Implemented from the paper:
# U-Net: Convolutional Networks for Biomedical
# Image Segmentation
# Olaf Ronneberger, Philipp Fischer, and Thomas Brox
# Computer Science Department and BIOSS Centre for Biological Signalling Studies,
# University of Freiburg, Germany
# https://arxiv.org/pdf/1505.04597.pdf


INPUT_WIDTH = 608
INPUT_HEIGHT = 416

OUTPUT_WIDTH = 304
OUTPUT_HEIGHT = 208

N_CLASSES = 7
N_EPOCHS = 100
STEP_PER_EPOCH = 512
BATCH_SIZE = 2


def u_net(input_height, input_width, output_height, output_width, n_classes):
    """
    Initialises a U-NET with Keras layers.
    Based on keras-segmentaion package
    https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py.

    :param input_height: the input image height
    :param input_width: the input image width
    :param n_classes: the number of classes for semantic segmentation
    :param pretrained_weights: by default None, if specified loads the weights from a file.
    :return: a Keras U-NET model.
    """

    # Going down in resolution, increasing feature channels
    img_input, levels = get_vgg_encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    x = f4

    x = layers.ZeroPadding2D((1, 1), data_format='channels_last')(x)
    x = layers.Conv2D(512, (3, 3), padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    # Going un in resolution, decreasing feature channels

    x = layers.UpSampling2D((2, 2), data_format='channels_last')(x)
    x = layers.concatenate((x, f3), axis=3)
    x = layers.ZeroPadding2D((1,  1), data_format='channels_last')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2), data_format='channels_last')(x)
    x = layers.concatenate((x, f2), axis=3)
    x = layers.ZeroPadding2D((1,  1), data_format='channels_last')(x)
    x = layers.Conv2D(128, (3, 3), padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2), data_format='channels_last')(x)
    x = layers.concatenate((x, f1), axis=3)
    x = layers.ZeroPadding2D((1,  1), data_format='channels_last')(x)
    x = layers.Conv2D(64, (3, 3), padding='valid', data_format='channels_last')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(n_classes, (1, 1), padding='same', data_format='channels_last')(x)
    x = layers.Reshape((output_height * output_width, -1))(x)
    output = layers.Activation('softmax')(x)

    return keras.Model(inputs=[img_input], outputs=[output])


def train_unet(input_dir, label_dir, epochs, save_path=None, plot=None):

    model = u_net(INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, N_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if plot is not None:
        plot_model(model, to_file=plot, show_shapes=True, show_layer_names=True)

    generators = get_default_datagen(input_dir, label_dir, (INPUT_WIDTH, INPUT_HEIGHT),
                                     (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    callback_list = []
    if save_path is not None:
        # Checkpoints, in the case the training is stopped, we can retrieve the weights from last epoch
        save_weights_callback = SaveWeightCallback(save_path)
        callback_list.append(save_weights_callback)

        # Saves all epoch info into a csv file, appends the data at each epoch.
        csv_logger = keras.callbacks.CSVLogger(os.path.join(save_path, 'log.csv'), separator=',', append=True)
        callback_list.append(csv_logger)

    print_lr_callback = PrintLR()
    callback_list.append(print_lr_callback)

    model.fit(x=generators, epochs=epochs, steps_per_epoch=STEP_PER_EPOCH, callbacks=callback_list)

    return model


if __name__ == '__main__':
    model_save_path = os.path.join(MODELS_DIRECTORY, UNET_CHECPOINT_PATH_SIMPLE)
    trained_model = train_unet(ORIGINAL_IMAGES_DIRECTORY, MERGED_LABEL_DIRECTORY, N_EPOCHS, save_path=model_save_path)

from keras.utils.vis_utils import plot_model
from src.u_net.unet_datagen import get_default_datagen, UnetDataGenerator
from src.global_var import *
from src.utils.custom_callbacks import SaveWeightCallback, PrintLR
from keras_contrib.callbacks import CyclicLR
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
    :param output_height: the output image height
    :param output_width: the output image width
    :param n_classes: the number of classes for semantic segmentation
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


def train_unet(input_dir, label_dir, epochs, optimizer=tf.keras.optimizers.Adadelta(), save_path=None, plot=None, additional_callbacks=[], input_val_dir=None, label_val_dir=None):

    validate = input_val_dir is not None and label_val_dir is not None

    model = u_net(INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, N_CLASSES)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if plot is not None:
        plot_model(model, to_file=plot, show_shapes=True, show_layer_names=True)

    # Data generator, performs image augmentation and loads image in memory only when needed.
    generators = UnetDataGenerator(input_dir, label_dir, (INPUT_WIDTH, INPUT_HEIGHT), (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                                   classes=N_CLASSES, batch_size=BATCH_SIZE, shuffle=True, repeats=3)

    validation_generator = None
    if validate:
        validation_generator = UnetDataGenerator(input_val_dir, label_val_dir, (INPUT_WIDTH, INPUT_HEIGHT),
                                                 (OUTPUT_WIDTH, OUTPUT_HEIGHT), classes=N_CLASSES, batch_size=BATCH_SIZE,
                                                 shuffle=True, repeats=12, validation=True)

    callback_list = []
    if save_path is not None:
        # Checkpoints, in the case the training is stopped, we can retrieve the weights from last epoch
        save_weights_callback = SaveWeightCallback(save_path)
        callback_list.append(save_weights_callback)

        # Saves all epoch info into a csv file, appends the data at each epoch.
        csv_logger = keras.callbacks.CSVLogger(os.path.join(save_path, 'log.csv'), separator=',', append=True)
        callback_list.append(csv_logger)
    callback_list += additional_callbacks

    # Prints the set learning rate, useful when using a Cyclic learning rate
    print_lr_callback = PrintLR()
    callback_list.append(print_lr_callback)

    model.fit(x=generators, epochs=epochs, steps_per_epoch=STEP_PER_EPOCH, callbacks=callback_list,
              validation_data=validation_generator, validation_batch_size=2, validation_freq=2)

    return model


if __name__ == '__main__':
    model_save_path = os.path.join(MODELS_DIRECTORY, UNET_CHECKPOINT_PATH_SIMPLE_VERIFICATION)
    images_path = os.path.join(ORIGINAL_IMAGES_DIRECTORY, 'img')
    label_path = os.path.join(MERGED_LABEL_DIRECTORY, 'img')
    images_val_dir = os.path.join(VERIFICATION_IMAGES_DIRECTORY, 'img')
    label_val_dir = os.path.join(VERIFICATION_LABEL_DIRECTORY, 'img')

    # cyclic_lr = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=1024, mode='triangular')

    trained_model = train_unet(images_path,
                               label_path,
                               N_EPOCHS,
                               save_path=model_save_path,
                               optimizer=tf.keras.optimizers.Adadelta(),
                               # additional_callbacks=[cyclic_lr],
                               input_val_dir=images_val_dir,
                               label_val_dir=label_val_dir)

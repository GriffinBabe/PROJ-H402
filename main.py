import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from tensorflow import keras
from preprocessing import rescale_images, flip_images

ORIGINAL_IMAGES_DIRECTORY = './dataset/original_images/'
LABELED_IMAGES_DIRECTORY = './dataset/label_images_semantic/'
PROCESSED_IMAGES_DIRECTORY = './dataset/processed_images/images/'
PROCESSED_LABELED_IMAGES_DIRECTORY = './dataset/processed_images/label/'
MODELS_DIRECTORY = './dataset/models/'
OUTPUT_DIRECTORY = './dataset/out/'

n_classes = 23  # all the classes we want to discriminate
n_epochs = 5


def train_vgg_model(input_image_dir, label_image_dir, input_w, input_h, epochs, number_classes):
    model = vgg_unet(n_classes, input_height=input_h, input_width=input_w)

    model.train(
        train_images=input_image_dir,
        train_annotations=label_image_dir,
        checkpoints_path='vgg_unet',
        epochs=epochs
    )
    return model


def preprocess_images_and_labels():
    rescale_images('dataset/original_images', 'dataset/processed_images', (1200, 800))
    flip_images('dataset/processed_images', 'dataset/processed_images', image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images', 'dataset/processed_images', image_suffix='_vflipped', horizontal=False)
    rescale_images('dataset/label_images_semantic', 'dataset/processed_images/label', (1200, 800))
    flip_images('dataset/processed_images/label', 'dataset/processed_images/label', image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images/label', 'dataset/processed_images/label', image_suffix='_vflipped', horizontal=False)


def load_model(path):
    model = keras.models.load_model(path)
    return model


# vgg_model = train_vgg_model(ORIGINAL_IMAGES_DIRECTORY, LABELED_IMAGES_DIRECTORY, 608, 416, n_epochs, n_classes)
# vgg_model.save(MODELS_DIRECTORY+"vgg_unet.model")
# vgg_model = load_model(MODELS_DIRECTORY+"vgg_unet.model")
# input_image = ORIGINAL_IMAGES_DIRECTORY+'000.jpg'
# prediction = vgg_model.predict(input_image)


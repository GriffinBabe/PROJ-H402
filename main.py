import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras_segmentation.models.unet import vgg_unet
from preprocessing import rescale_images, flip_images, split_images
from statistics import select_complex_labels
import pickle

ORIGINAL_IMAGES_DIRECTORY = './dataset/original_images/'
LABELED_IMAGES_DIRECTORY = './dataset/label_images_semantic/'
PROCESSED_IMAGES_DIRECTORY = './dataset/processed_images/images_splitted/'
PROCESSED_LABELED_IMAGES_DIRECTORY = './dataset/processed_images/label_splitted/'
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
    rescale_images('dataset/original_images',
                   'dataset/processed_images/images_rescaled',
                   (5632, 3584))
    flip_images('dataset/processed_images/images_splitted',
                'dataset/processed_images/images_splitted',
                image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images',
                'dataset/processed_images',
                image_suffix='_vflipped', horizontal=False)
    rescale_images('dataset/label_images_semantic',
                   'dataset/processed_images/label_rescaled',
                   (5632, 3584))
    flip_images('dataset/processed_images/label_splitted',
                'dataset/processed_images/label_splitted',
                image_suffix='_flipped', horizontal=True)
    flip_images('dataset/processed_images/label_splitted',
                'dataset/processed_images/label_splitted',
                image_suffix='_vflipped', horizontal=False)


# model = train_vgg_model('dataset/processed_images/images_splitted',
#                         'dataset/processed_images/label_splitted',
#                         512, 512, 20, 23)
#
# input_image_1 = 'dataset/processed_images/images_splitted/000_s0_0.jpg'
# input_image_2 = 'dataset/processed_images/images_splitted/000_s0_2.jpg'
# input_image_3 = 'dataset/processed_images/images_splitted/004_s0_2.jpg'
#
# model.predict_segmentation(inp=input_image_1, out_fname=OUTPUT_DIRECTORY+'out1.png')
# model.predict_segmentation(inp=input_image_2, out_fname=OUTPUT_DIRECTORY+'out2.png')
# model.predict_segmentation(inp=input_image_3, out_fname=OUTPUT_DIRECTORY+'out3.png')

select_complex_labels('dataset/processed_images/label_splitted/',
                      label_output_dir='dataset/processed_images/label_selected',
                      image_dir='dataset/processed_images/images_splitted',
                      image_output_dir='dataset/processed_images/images_selected',
                      min_types=5)

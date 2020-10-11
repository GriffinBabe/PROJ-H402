import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras_segmentation.models.unet import vgg_unet

ORIGINAL_IMAGES_DIRECTORY = "./dataset/original_images/"
LABELED_IMAGES_DIRECTORY = "./dataset/label_images_semantic/"
MODELS_DIRECTORY = "./dataset/models/"

n_classes = 23  # all the classes we want to discriminate


def train_vgg_model(input_image_dir, label_image_dir, input_w, input_h, epochs, number_classes):
    model = vgg_unet(n_classes, input_height=input_h, input_width=input_w)

    model.train(
        train_images=ORIGINAL_IMAGES_DIRECTORY,
        train_annotations=LABELED_IMAGES_DIRECTORY,
        checkpoints_path="vgg_unet",
        epochs=epochs
    )
    return model


vgg_model = train_vgg_model(ORIGINAL_IMAGES_DIRECTORY, LABELED_IMAGES_DIRECTORY, 608, 416, 20, n_classes)
vgg_model.save(MODELS_DIRECTORY+"vgg_unet.model")

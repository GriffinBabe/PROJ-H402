from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import ModelCheckpoint
from src.global_var import *
import os

input_width = 608
input_height = 416

output_width = 304
output_height = 208

n_classes = 23  # all the classes we want to discriminate
n_epochs = 100


def train_vgg_model(weight_save, epochs=n_epochs, checkpoints_path=None):
    model = vgg_unet(n_classes=n_classes, input_height=input_height, input_width=input_width)
    model.train(train_images=ORIGINAL_IMAGES_DIRECTORY,
                train_annotations=LABELED_IMAGES_DIRECTORY,
                checkpoints_path=checkpoints_path,
                epochs=epochs)

    model.save_weights(os.path.join(MODELS_DIRECTORY, weight_save))
    return model


if __name__ == '__main__':
    checkpt_paths = os.path.join(MODELS_DIRECTORY, VGG_CHECKPOINT_PATH)
    model = train_vgg_model(VGG16_5_EPOCHS, epochs=n_epochs, checkpoints_path=checkpt_paths)

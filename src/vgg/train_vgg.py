from keras_segmentation.models.unet import vgg_unet
from src.global_var import *
import os

input_width = 608
input_height = 416

output_width = 304
output_height = 208

n_classes = 7  # all the classes we want to discriminate
n_epochs = 60


def train_vgg_model(weight_save, checkpoints_path, epochs=n_epochs):
    model = vgg_unet(n_classes=n_classes, input_height=input_height, input_width=input_width)
    # model.load_weights(os.path.join(checkpt_paths, '.7'))

    model.train(train_images=os.path.join(ORIGINAL_IMAGES_DIRECTORY, 'img'),
                train_annotations=os.path.join(MERGED_LABEL_DIRECTORY, 'img'),
                auto_resume_checkpoint=True,
                checkpoints_path=checkpoints_path,
                verify_dataset=False,
                epochs=epochs)

    model.save_weights(os.path.join(MODELS_DIRECTORY, weight_save))
    return model


if __name__ == '__main__':
    checkpt_paths = os.path.join(MODELS_DIRECTORY, VGG_CHECKPOINT_PATH_SIMPLE)
    model = train_vgg_model(VGG16_60_EPOCHS_SIMPLE, checkpoints_path=checkpt_paths,  epochs=n_epochs)

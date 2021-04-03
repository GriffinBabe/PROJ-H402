from keras_segmentation.models.unet import vgg_unet
from keras_contrib.callbacks import CyclicLR
from src.global_var import *
from src.utils.custom_callbacks import PrintLR
import os

input_width = 608
input_height = 416

output_width = 304
output_height = 208

n_classes = 7  # all the classes we want to discriminate
n_epochs = 100

def train_vgg_model(weight_save, checkpoints_path, epochs=n_epochs):
    model = vgg_unet(n_classes=n_classes, input_height=input_height, input_width=input_width)
    # model.load_weights(os.path.join(checkpt_paths, '.38'))

    # Set a cyclical learning rate as a callback. The learning rate varies between 0.005 and 0.0001
    # A triangle shape of cycle is used
    cyclic_lr = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=1024, mode='triangular')
    print_lr = PrintLR()

    model.train(train_images=os.path.join(AUGMENTED_IMAGES_DIRECTORY, 'img'),
                train_annotations=os.path.join(AUGMENTED_LABEL_DIRECTORY, 'img'),
                auto_resume_checkpoint=True,
                checkpoints_path=checkpoints_path,
                verify_dataset=False,
                epochs=epochs,
                more_callbacks=[cyclic_lr, print_lr])

    model.save_weights(os.path.join(MODELS_DIRECTORY, weight_save))
    return model


if __name__ == '__main__':
    checkpt_paths = os.path.join(MODELS_DIRECTORY, VGG_CHECKPOINT_PATH_SIMPLE)
    model = train_vgg_model(VGG16_60_EPOCHS_SIMPLE, checkpoints_path=checkpt_paths,  epochs=n_epochs)

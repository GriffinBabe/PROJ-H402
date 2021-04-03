# Visualization script of the ImageDataGenerator output for debugging
import matplotlib.pyplot as plt
from src.u_net.unet_datagen import get_default_datagen
from src.global_var import *

INPUT_WIDTH = 608
INPUT_HEIGHT = 416

OUTPUT_WIDTH = 304
OUTPUT_HEIGHT = 208

N_CLASSES = 7
N_EPOCHS = 100
STEP_PER_EPOCH = 512
BATCH_SIZE = 2


if __name__ == '__main__':
    input_dir = ORIGINAL_IMAGES_DIRECTORY
    label_dir = MERGED_LABEL_DIRECTORY
    generators = get_default_datagen(input_dir, label_dir, (INPUT_WIDTH, INPUT_HEIGHT),
                                     (OUTPUT_WIDTH, OUTPUT_HEIGHT), batch_size=BATCH_SIZE)

    while True:
        input_image_batch, label_image_batch = generators.__next__()
        print(label_image_batch[0].shape)
        figure = plt.figure(figsize=(8, 8))
        plt.subplot(221)
        plt.imshow(input_image_batch[0].transpose(1, 0, 2))
        plt.subplot(222)
        plt.imshow(label_image_batch[0].transpose(1, 0, 2))
        plt.subplot(223)
        plt.imshow(input_image_batch[1].transpose(1, 0, 2))
        plt.subplot(224)
        plt.imshow(label_image_batch[1].transpose(1, 0, 2))
        plt.show()

# Visualization script of the ImageDataGenerator output for debugging
import matplotlib.pyplot as plt
from src.u_net.unet_datagen import get_default_datagen, UnetDataGenerator
from src.global_var import *
import os

INPUT_WIDTH = 608
INPUT_HEIGHT = 416

OUTPUT_WIDTH = 304
OUTPUT_HEIGHT = 208

N_CLASSES = 7
N_EPOCHS = 100
STEP_PER_EPOCH = 512
BATCH_SIZE = 2


if __name__ == '__main__':
    input_dir = os.path.join(ORIGINAL_IMAGES_DIRECTORY, 'img')
    label_dir = os.path.join(MERGED_LABEL_DIRECTORY, 'img')
    generators = UnetDataGenerator(input_dir, label_dir, (INPUT_WIDTH, INPUT_HEIGHT), (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                                   classes=N_CLASSES, batch_size=BATCH_SIZE, shuffle=True, repeats=3)

    while True:
        input_image_batch, label_image_batch = generators[0]
        print(label_image_batch[0].shape)
        figure = plt.figure(figsize=(8, 8))
        plt.subplot(221)
        plt.imshow(input_image_batch[0])
        plt.subplot(222)
        plt.imshow(label_image_batch[0])
        plt.subplot(223)
        plt.imshow(input_image_batch[1])
        plt.subplot(224)
        plt.imshow(label_image_batch[1])
        plt.show()

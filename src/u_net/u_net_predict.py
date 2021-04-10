from src.global_var import *
from src.u_net.u_net import u_net, INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_WIDTH, OUTPUT_HEIGHT, N_CLASSES
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
import os
import cv2
import numpy as np
import time


def decode_prediction(image):
    output = np.zeros(image.shape[0], np.int)
    for idx in range(image.shape[0]):
        max = np.argmax(image[idx])
        output[idx] = max + 1
    return output.reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Arguments expected: python u_net_predict.py <input_image_number>')

    image_path = os.path.join(ORIGINAL_IMAGES_DIRECTORY, 'img', sys.argv[1]+'.jpg')
    label_path = os.path.join(MERGED_LABEL_DIRECTORY, 'img', sys.argv[1]+'.png')

    model = u_net(INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, N_CLASSES)
    model.load_weights(os.path.join(MODELS_DIRECTORY, UNET_CHECKPOINT_PATH_SIMPLE, '41.weights'))

    augmenter = ImageDataGenerator(rotation_range=30,
                                   zoom_range=0.2,
                                   width_shift_range=0.01,
                                   height_shift_range=0.01,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   samplewise_std_normalization=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), cv2.INTER_NEAREST).astype(np.float64) / 255.0
    image = augmenter.standardize(image)

    input_batch = np.array([image])

    output = model.predict(input_batch, batch_size=1)

    segmentation = decode_prediction(output[0])

    figure = plt.figure(figsize=(10, 8))

    plt.subplot(211)
    plt.imshow(segmentation)
    plt.subplot(212)
    plt.imshow(label)
    plt.show()

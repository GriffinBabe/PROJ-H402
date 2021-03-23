from src.global_var import *
from src.vgg.train_vgg import n_classes, input_width, input_height, output_width, output_height
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_segmentation.models.unet import vgg_unet
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os

# Color linspace for segmentation output presentation
t = np.linspace(-510, 510, n_classes)
color_array = np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)


def load_vgg16_model(model_path):
    model = vgg_unet(n_classes, input_width=input_width, input_height=input_height)
    # model.build((input_width, input_height))
    model.load_weights(model_path)
    return model


def process_input(image):
    image = resize(image, (input_height, input_width))
    # Deep learning models expect batches of images, that's why we add a dimension
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)


def process_output(output):
    output_list = []
    for out in output:
        out = out.reshape((output_height, output_width, n_classes))
        # Merge each channel into an image
        out_processed = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        out_processed[:, :] = color_array[np.argmax(out, axis=2)]
        output_list.append(out_processed)
    return output_list


if __name__ == '__main__':
    input_image = imread(os.path.join(ORIGINAL_IMAGES_DIRECTORY, 'img', '100.jpg'))
    label_image = imread(os.path.join(LABELED_IMAGES_DIRECTORY, 'img', '100.png'))
    model = load_vgg16_model(os.path.join(MODELS_DIRECTORY, VGG16_60_EPOCHS))

    output_path = os.path.join(OUTPUT_DIRECTORY, '100.png')
    out = model.predict_segmentation(inp=input_image, out_fname=output_path)

    plt.subplot(211)
    plt.imshow(out)
    plt.subplot(212)
    plt.imshow(label_image)
    plt.show()


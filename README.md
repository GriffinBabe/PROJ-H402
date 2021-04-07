# PROJ-H402
Repository for the 2020-2021 computing project @ Ecole Polytechnique de Bruxelles

# Installation
Required packages: Tensorflow, Keras, Numpy, Scipy, Matplotlib, Scikit-image, keras-contrib, keras-segmentation

Dowload the dataset from https://www.kaggle.com/bulentsiyah/semantic-drone-dataset and put the data in a "dataset" folder at the root folder of the repository.

The labels need to be merged with the `src/preprocessing/merge_classes.py` script.

Prepare the dataset folders by adding an extra 'img' folder. The labels need to be merged with the `src/preprocessing/merge_classes.py` script.

For example, input image 001 will be searched with the path `dataset/original_images/img/001.jpg`.
While its label will be searched with the path `dataset/label_images_merged/img/001.png`.


Image augmentation is done during the training with the `UnetDataGenerator` data generator defined in the `src/u_net/unet_datagen.py` file.

# Training

An utilisation of the library `keras-segmentation` can be found in the script `src/vgg/train_vgg.py`.

A custom network was written in the script `src/u_net/u_net.py`.



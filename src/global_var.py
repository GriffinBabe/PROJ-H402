ORIGINAL_IMAGES_DIRECTORY = './dataset/original_images/'
LABELED_IMAGES_DIRECTORY = './dataset/label_images_semantic/'

MERGED_LABEL_DIRECTORY = './dataset/label_images_merged/'

AUGMENTED_IMAGES_DIRECTORY = './dataset/original_images_augmented'
AUGMENTED_LABEL_DIRECTORY = './dataset/label_images_augmented/'

VERIFICATION_IMAGES_DIRECTORY = './dataset/original_images_verification/'
VERIFICATION_LABEL_DIRECTORY = './dataset/label_images_merged_verification/'

MODELS_DIRECTORY = './models/'
OUTPUT_DIRECTORY = './out/'

# Full 23 classes
VGG_CHECKPOINT_PATH = 'vgg_unet_checkpoint/'
VGG16_60_EPOCHS = 'vgg_unet_60_epochs/vgg'

# Reduced to 7 classes for easier task
VGG_CHECKPOINT_PATH_SIMPLE = 'vgg_unet_checkpoint_simple/'
VGG16_60_EPOCHS_SIMPLE = 'vgg_unet_60_epochs_simple/vgg'

# Reduced to 7 classes U-net
UNET_CHECKPOINT_PATH_SIMPLE = 'unet_checkpoint_simple/'
UNET_CHECKPOINT_PATH_SIMPLE_VERIFICATION = 'unet_checkpoint_simple_verification/'

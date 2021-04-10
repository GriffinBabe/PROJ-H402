import os
import hashlib
import sys


def split_dataset(input_dir, label_dir, input_ver_dir, label_ver_dir):
    images_to_verification = []

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        hs = bytes(hashlib.sha256(file.encode('utf-8')).hexdigest(), encoding='raw_unicode_escape')
        first_byte = hs[0]
        # Look if the last 2 bits are not two ones, around 1/4 of the dataset is splitted
        if not first_byte & 0x03:
            images_to_verification.append(file)

    label_to_verification = [i.replace('.jpg', '.png') for i in images_to_verification]
    images_labels_to_verification = zip(images_to_verification, label_to_verification)

    for img, label in images_labels_to_verification:
        os.rename(os.path.join(input_dir, img), os.path.join(input_ver_dir, img))
        os.rename(os.path.join(label_dir, label), os.path.join(label_ver_dir, label))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise Exception('Arguments expected: python verification_split.py '
                        '<input_dir> <label_dir> <output_dir> <output_label_dir>')

    input_dir = sys.argv[1]
    label_dir = sys.argv[2]
    input_ver_dir = sys.argv[3]
    label_ver_dir = sys.argv[4]

    split_dataset(input_dir, label_dir, input_ver_dir, label_ver_dir)

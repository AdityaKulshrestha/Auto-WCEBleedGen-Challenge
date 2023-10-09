import glob
import random
import os
import shutil

PATH = 'Dataset/WCEBleedGen/bleeding/'
img_path = glob.glob(PATH + 'images/*.png')
labels = glob.glob(PATH + 'Bounding boxes/YOLO_TXT/*.txt')


def move(paths, folder):
    for p in paths:
        shutil.move(p, folder)


def split_dataset_yolo(img_path: str, labels: str, ratio: int = 0.8):
    """
    Splits the dataset into training and testing
    :param img_path: Path to the images
    :param labels: Path to the labels
    :param ratio: Ratio of training to validation
    :return: None
    """
    # Calculate number of files for training and validation
    data_size = len(img_path)
    train_size = int(data_size * ratio)

    # Shuffle two list
    img_labels = list(zip(img_path, labels))
    random.seed(43)
    random.shuffle(img_labels)
    img_paths, txt_paths = zip(*img_labels)

    # Splitting
    train_img_paths = img_paths[:train_size]
    train_txt_paths = txt_paths[:train_size]

    valid_img_paths = img_paths[train_size:]
    valid_txt_paths = txt_paths[train_size:]

    train_folder = 'segmentation/dataset/train'
    valid_folder = 'segmentation/dataset/valid'

    split = ['/images', '/labels']

    for type in split:
        os.makedirs(train_folder + type)
        os.makedirs(valid_folder + type)

    move(train_img_paths, train_folder + '/images')
    move(train_txt_paths, train_folder + '/labels')
    move(valid_img_paths, valid_folder + '/images')
    move(valid_txt_paths, valid_folder + '/labels')


if __name__ == '__main__':
    split_dataset_yolo(img_path, labels)

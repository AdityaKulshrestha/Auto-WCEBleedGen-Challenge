import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from configs.training_config import DATASET_PATH
from classification.dataloader import load_dataset, split_dataset
from configs.training_config import STD, MEAN

# For solving QT related issue locally
import matplotlib
matplotlib.use('TKAgg')


def invert_transform(images: torch.Tensor):
    """
    Displayes the images in grid without labels
    :param images: Batch from Images dataset
    :return: None
    """
    images = torchvision.utils.make_grid(images)
    images = images * STD + MEAN
    npimg = images.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_images(images: torch.Tensor, labels: torch.Tensor, rows: int = 4, cols: int = 4):
    """
    Plots the images in a grid.
    :param images: Single batch of images
    :param labels: Corresponding labels of the images
    :param rows: Number of rows
    :param cols: Number of columns
    :return: None
    """
    img_count = 0
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))
    # images = images * STD[:, None, None] + MEAN[:, None, None]
    for i in range(rows):
        for j in range(cols):
            if img_count < len(images):
                axes[i, j].imshow(np.transpose(images[img_count], (1, 2, 0)))
                axes[i, j].set_title(labels[img_count])
                img_count += 1
    plt.show()


if __name__ == "__main__":
    dataset, classes_labels = load_dataset(DATASET_PATH)
    training_data, validation_data = split_dataset(dataset, 0.8)
    imgs, lbls = next(iter(validation_data))
    plot_images(imgs, lbls)
    # invert_transform(imgs)

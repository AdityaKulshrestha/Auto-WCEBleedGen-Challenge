import torch
from torch.utils.data import Dataset
from configs.training_config import DATASET_PATH, BATCH_SIZE
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
from torch.utils.data import DataLoader, random_split

def load_dataset(path: str = '/kaggle/working/dataset/') -> datasets:
    """
    Loads the data from the given directory
    :param path: Path of the dataset in structured forlder for each class.
    :return: Pytorch.datasets
    """
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(path, transform=data_transforms)

    classesLabels = dataset.class_to_idx
    return dataset, classesLabels


# Splitting the data into training and testing dataset
def split_dataset(dataset: datasets, split_size: int = 0.8):
    """
    Splits the dataset into training and testing data
    :param split_size: Ratio of train to validation size
    :param dataset: Original dataset
    :return: Training and testing dataset
    """
    train_size = int(0.8*len(dataset))
    valid_size = len(dataset) - train_size

    # Splitting the dataset
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader




import torch
import os
from configs.training_config import DATASET_PATH, NUM_CLASSES, EPOCHS, MODEL_DIR
from torchvision import models
from classification.dataloader import load_dataset, split_dataset
import torch.nn as nn
import torch.optim as optim


class EfficientModel(nn.Module):
    """
    Transfer Learning using Efficient b0 Model
    """

    def __init__(self, num_classes: int):
        super(EfficientModel, self).__init__()

        # Load the pre-trained EfficientNet -b0 model
        self.model = models.efficientnet_b0(pretrained=True)

        # Replace the classifier with a new one
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train(train_dataset, validation_dataset):
    """
    Function for training on the dataset
    :param train_dataset: Training Data (images, labels)
    :param validation_dataset: Validation Data (images, labels)
    :return: None
    """
    # Set device to GPU for faster training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Creating an instance of the EfficientModel
    model = EfficientModel(NUM_CLASSES)
    model.to(device)
    # Defining Loss function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculation Validation Loss
        val_loss = 0.0
        for i, data in enumerate(validation_dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        print(
            f'Epoch {epoch + 1}/{EPOCHS} \t Training Loss: {running_loss / len(train_dataset)} \t Validation Loss: {val_loss / len(validation_dataset)}')

    print("Training Complete")

    # checkpoint = {
    #     'transfer_model': "EfficientNet",
    #     'features': model.features,
    #     'classifier': model.classifier,
    #     'optimizer': optimizer.state_dict(),
    #     'idx_to_class': model.class_to_idx
    # }

    # Create a directory for saving a model if it doesn't exists
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    torch.save(model.state_dict(), MODEL_DIR+ '/model.pth')
    print("Model Saved To :" + MODEL_DIR+'/model.pth')


if __name__ == "__main__":
    dataset, classes_labels = load_dataset(DATASET_PATH)
    training_data, validation_data = split_dataset(dataset, 0.8)
    train(training_data, validation_data)

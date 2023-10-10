# AutoWCEBleeding Challenge

## Introduction

Welcome to my submission for the Auto-WCEBleedGen challenge. The goal is to develop and evaluate an AI model for the automatic detection and classification of bleeding and non-bleeding frames in Wireless Capsule Endoscopy (WCE) Images.


## Repository Contents

### Flow
The directory contains two sub-directory for the classification and segmentation.
- First the image is classified into bleeding and non-bleeding image. 
- Then, the trained model is used to detect the bleeding region within the bleeding image.

### Code
The code is divided into two submodules: classification and segmentation.
#### Classification
- `train.py`: Python script for training our AI model.
- `validate.py`: Python script for validating our AI model.
- `test.py`: Python script for testing our AI model.
- `utils/`: Directory containing utility scripts and helper functions.
- `config/`: Directory containing configuration files.
- `checkpoints/`: Directory to store model checkpoints (optional).
- `assets/`: Directory for additional assets and resources.
- `README.md`: This README file.
#### Segmentation

### Model

- [Include details about your trained model here, if applicable.]

### Excel Sheet

- `predictions.xlsx`: Excel sheet containing image IDs and predicted class labels for testing dataset 1 and 2.

## Usage

- Classification
  - Configure the config file. 
  - Prepare the data by running the data_preparation.py file.
  - Use train.py to train the model on the data. 
  - Next, evaluate the performance using evaluate.py file.

## Evaluation Metrics

Here are the evaluation metrics for the model:

- Classification

| Metric                 | Classification |
|------------------------|----------------|
| Accuracy               | [Accuracy]     |
| Recall                 | [Recall]       |
| F1-Score               | [F1-Score]     |

- Detection 

| Metric                      | Value |
|-----------------------------|-------|
| Average Precision (AP)      | [AP]  |
| Mean Average Precision (mAP)| [mAP] |
| Intersection over Union (IoU)| [IoU] |
## Results



### Testing Dataset 1 & 2

![Image 1](assets/testing_image_1.png)
![Image 2](assets/testing_image_2.png)
![Image 3](assets/testing_image_3.png)


![Image 4](assets/testing_image_4.png)
![Image 5](assets/testing_image_5.png)
![Image 6](assets/testing_image_6.png)


## Future Work / To-Do List

- ~~Segregate data~~
- ~~Add training scripts.~~
- ~~Add configuration file.~~
- Add evaluation scripts. 
- Add visualization script
- Develop pipeline for inference. 
- Update the readme.
- Add images and final compiled result.

## Questions or Feedback

If you have any questions or feedback, please feel free to contact us.

---

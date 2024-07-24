# Skin Disease Predictor

## Overview

This notebook demonstrates a neural network-based approach to predict skin diseases using a Convolutional Neural Network (CNN) with Transfer Learning. By leveraging a pre-trained InceptionV3 model, this script fine-tunes the model to classify skin disease images from the DermNet dataset. This method allows for accurate disease prediction while minimizing the need for extensive training data.

## Key Features

- **Transfer Learning**: Utilizes the InceptionV3 model pre-trained on ImageNet to extract features from skin disease images, followed by custom layers for classification.
- **Data Augmentation**: Implements various data augmentation techniques to enhance model generalization and performance.
- **Model Checkpointing**: Saves the best-performing model based on validation loss during training for later evaluation and inference.
- **High Accuracy**: Achieves impressive accuracy on the validation and test datasets.

## Setup Instructions

### Prerequisites

To run this project, you need to install the following libraries:

- `keras`
- `tensorflow`
- `numpy`
- `glob`

### Installation

You can install the required libraries using `pip`. Run the following commands in your terminal or command prompt:

```bash
pip install keras
pip install tensorflow
pip install numpy
pip install glob
```

### Dataset

The dataset used for this project is available at the following link:

[DermNet Dataset](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)

Please download the dataset and ensure it is properly structured with the training and test images organized into respective folders.

# Eye Disease Classification Using CNN, VGG16, and ResNet-50

## Overview

This repository contains the implementation of an eye disease classification system using Convolutional Neural Networks (CNN), VGG16, and ResNet-50 architectures. The project aims to accurately classify various eye diseases from retinal images.

This project is part of the GirlScript Summer of Code 2024 program.

## Table of Contents
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [License](#license)

## Features
- Data preprocessing and augmentation.
- Implementation of CNN, VGG16, and ResNet-50 models.
- Training and validation scripts.

## Dataset
The dataset used for this project consists of retinal images labeled with various eye diseases. You can download the dataset from [Kaggle Eye Disease Dataset](https://www.kaggle.com/datasets/kondwani/eye-disease-dataset).

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/eye-disease-classification.git
    cd eye-disease-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Preprocess the Data:**
    - Download the dataset and place it in the `data/` directory.
    - Run the preprocessing script to prepare the data for training:
        ```bash
        python preprocess.py
        ```

2. **Train the Model:**
    - Train the CNN model:
        ```bash
        python train.py --model cnn
        ```
    - Train the VGG16 model:
        ```bash
        python train.py --model vgg16
        ```
    - Train the ResNet-50 model:
        ```bash
        python train.py --model resnet50
        ```

3. **Evaluate the Model:**
    - Evaluate the trained model:
        ```bash
        python evaluate.py --model model_name
        ```

## Model Architecture
### Convolutional Neural Network (CNN)
A custom CNN architecture designed for image classification.

### VGG16
A pre-trained VGG16 model fine-tuned for eye disease classification.

### ResNet-50
A pre-trained ResNet-50 model fine-tuned for eye disease classification.

## Training
The training process involves:
- Splitting the dataset into training, validation, and test sets.
- Data augmentation to improve model generalization.
- Training the model with early stopping and model checkpointing.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

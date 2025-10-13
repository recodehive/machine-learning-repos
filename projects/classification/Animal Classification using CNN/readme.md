
# Animal Classifier using CNN

This project is an implementation of an Artificial Neural Network (ANN) to classify images of cats and dogs. The dataset used contains images of cats and dogs and is obtained from Kaggle. The implementation is done in Python using popular machine learning libraries.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Project Overview

The goal of this project is to classify images of cats and dogs using a Convolutional Neural Network (CNN). The model is trained on a dataset of images obtained from Kaggle. This project demonstrates the process of loading the dataset, preprocessing the images, building the model, training the model, and evaluating its performance.

## Dataset

The dataset used in this project is from Kaggle and contains 2500+ images of cats and dogs. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/salader/dogs-vs-cats).

## Installation

To run this project, you need to have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Kaggle API

You can install the necessary packages using the following commands:

```bash
pip install tensorflow keras numpy matplotlib kaggle
```

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/recodehive/machine-learning-repos/Animal Classification using CNN.git
   cd cat-vs-dog-classifier
   ```

2. **Download the dataset:**
   Ensure you have your Kaggle API credentials (`kaggle.json`) and run the following commands to download and unzip the dataset:
   ```bash
   mkdir -p ~/.kaggle
   cp path_to_your_kaggle.json ~/.kaggle/
   kaggle datasets download -d salader/dogs-vs-cats
   unzip dogs-vs-cats.zip -d ./data
   ```

3. **Run the Jupyter notebook:**
   Open the `animal_classification` notebook and run the cells to execute the code step-by-step.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Fully connected (Dense) layers
- Dropout layers to prevent overfitting
- Output layer with softmax activation for classification

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss. The dataset is split into training and validation sets to monitor the performance of the model during training.

## Evaluation

The model is evaluated on a separate test set to measure its accuracy and other performance metrics. The evaluation results, including accuracy and loss, are plotted using Matplotlib.

## Results

The model achieves an accuracy of approximately 95% on the test set. The training and validation accuracy and loss curves are plotted to visualize the model's performance.


# Fake News Detection using CNN

This project implements a Convolutional Neural Network (CNN) to detect and classify fake news articles from textual data. The dataset used consists of labeled news articles, categorized as either real or fake, and is obtained from various sources including Kaggle. The implementation is done in Python using popular machine learning libraries.

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
The goal of this project is to detect fake news using a Convolutional Neural Network (CNN). The model is trained on a dataset of news articles labeled as real or fake. This project demonstrates the process of loading the dataset, preprocessing the text, building the model, training the model, and evaluating its performance.

## Dataset
The dataset used in this project is from Kaggle and contains thousands of labeled news articles. You can download the dataset from [here](https://www.kaggle.com/c/fake-news).

## Installation
To run this project, you need to have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install the necessary packages using the following commands:

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib
```
Usage
Clone the repository:
```bash
git clone https://github.com/recodehive/machine-learning-repos/Fake-News-Detection.git
cd Fake-News-Detection
```
## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:
- Embedding layer for word representation
- Convolutional layers to capture spatial hierarchies
- MaxPooling layers to reduce dimensionality
- Flatten layer to convert the 3D output to 1D
- Fully connected (Dense) layers for classification
- Output layer with a sigmoid activation function for binary classification

## Training
The model is trained using the Adam optimizer and binary cross-entropy loss. The dataset is split into training and validation sets to monitor the performance of the model during training.

## Evaluation
The model is evaluated on a separate test set to measure its accuracy. The evaluation results, including accuracy and loss, are printed to the console.

## Results
The model achieves competitive accuracy on the test set. Training and validation accuracy can be visualized through plots generated during training.

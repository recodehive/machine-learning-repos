# Alzheimer's Disease Classification Using CNN

## Project Overview

This project aims to classify images of individuals into different categories of Alzheimer's disease using Convolutional Neural Networks (CNN). The dataset used includes images from four classes: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. The model is trained to recognize features that distinguish these classes, providing a tool for early diagnosis and research.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)

## Installation

To run this project, you'll need to have Python 3.x and the following packages installed:

```bash
pip install pandas numpy opencv-python matplotlib tensorflow imbalanced-learn
```

You can clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <repository-name>
```

### Dataset
The dataset used in this project is the Alzheimer’s Dataset, which contains images categorized into four classes:

- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented
The dataset can be downloaded from the following link: Alzheimer's Dataset.

### Model Architecture
The model is built using the Keras Sequential API. The architecture consists of:

- Input Layer: Input shape of (176, 176, 3)
- Flatten Layer: Converts the 2D image into a 1D array.
- Dense Layers: Five hidden layers with ReLU activation functions.
- Output Layer: Softmax activation function to predict class probabilities.

### Training
The model is trained using:

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: AUC (Area Under Curve)




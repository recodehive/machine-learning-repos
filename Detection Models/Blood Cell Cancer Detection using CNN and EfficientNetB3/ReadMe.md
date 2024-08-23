# Blood Cell Cancer Classification using CNN and EfficientNetB3

This project aims to classify blood cell images to detect cancerous cells using Convolutional Neural Networks (CNN) and EfficientNetB3 architecture.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Notebook Structure](#notebook-structure)
  - [Import Necessary Libraries](#import-necessary-libraries)
  - [Reading the Data](#reading-the-data)
  - [Explore the Data](#explore-the-data)
  - [Data Preprocessing](#data-preprocessing)
  - [Building the Model](#building-the-model)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Using EfficientNetB3](#using-efficientnetb3)
  - [Conclusion](#conclusion)
- [Results](#results)

## Overview
This project utilizes deep learning techniques to classify blood cell images into cancerous and non-cancerous categories. Initially, a basic CNN model is implemented, followed by an enhanced model using EfficientNetB3 architecture for improved accuracy.

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/) and contains images of various blood cell types. The dataset is organized into folders for each cell type.

## Installation
To run the notebook, ensure you have the following libraries installed:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- PIL

You can install the required libraries using:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python pillow
```

## Notebook Structure

### Import Necessary Libraries
This section imports all the required libraries for data handling, preprocessing, and building the CNN model.

### Reading the Data
The data is read from the specified directory, and file paths along with labels are stored in a DataFrame.

### Explore the Data
Exploratory data analysis is performed to understand the distribution and characteristics of the dataset.

### Data Preprocessing
Data preprocessing steps include splitting the dataset into training and validation sets and augmenting the images.

### Building the Model
A Convolutional Neural Network (CNN) model is built using Keras. The model architecture includes convolutional layers, pooling layers, and dense layers.

### Training the Model
The model is trained on the preprocessed dataset with specified parameters.

### Evaluating the Model
Model performance is evaluated using metrics such as confusion matrix and classification report.

### Using EfficientNetB3
EfficientNetB3 architecture is used to enhance the model's accuracy. The pre-trained EfficientNetB3 model is fine-tuned on the dataset.

### Conclusion
Summary of the findings and results, including insights on model performance and potential improvements.

## Results
The project demonstrates the capability of CNN and EfficientNetB3 in classifying blood cell images with high accuracy.

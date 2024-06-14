# Image Classification Model - README

## Overview

This project involves creating a deep learning model using TensorFlow and Keras to classify images into two categories: happy and sad people. The dataset is loaded from a directory, preprocessed, and then used to train, validate, and test a Convolutional Neural Network (CNN). The model is then used to predict the class of new images.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- TensorFlow
- OpenCV
- NumPy
- Matplotlib

You can install these libraries using pip:

```bash
pip install tensorflow opencv-python-headless numpy matplotlib
```

## Project Structure

- `data/`: Directory containing subdirectories of images, with each subdirectory representing a class.
- `sadtest.jpg`: An example image used for testing the model.
- `model.py`: The main Python script containing the code for preprocessing, training, and evaluating the model.

## Steps to Run the Project

### 1. Data Preparation

The script starts by preparing the data:

1. **Filter out invalid images:**  
   The script iterates through the dataset directory and removes any file that is not an image or has an unsupported extension.

2. **Load and preprocess the data:**  
   The dataset is loaded using `tf.keras.utils.image_dataset_from_directory` and then normalized.

### 2. Data Visualization

Visualize a batch of images to understand the dataset:

### 3. Data Splitting

Split the data into training, validation, and test sets:

### 4. Model Building

Build and compile the CNN model:

### 5. Model Training

Train the model using the training and validation sets:

### 6. Model Evaluation

Plot the loss and accuracy curves, and evaluate the model on the test set:

### 7. Prediction

Use the model to predict the class of a new image:

## Conclusion

This README provides a structured guide to understanding and running the image classification project. Follow the steps outlined to preprocess the data, build, train, and evaluate the model, and finally make predictions on new images.

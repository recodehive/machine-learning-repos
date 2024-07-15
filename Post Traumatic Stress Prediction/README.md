## Overview 
This project utilizes a Convolutional Neural Network (CNN) to detect Post-Traumatic Stress Disorder (PTSD) from textual data. The model is trained on a dataset containing text samples from individuals diagnosed with PTSD and those without the condition.

## Dataset
The dataset is divided into two categories:

with_ptsd: Text samples from individuals diagnosed with PTSD.
<br>
without_ptsd: Text samples from individuals without PTSD.

## Model Architecture
The model consists of the following layers:

- Embedding layer for text data
- Convolutional layers with ReLU activation and MaxPooling
- Flatten layer
- Fully connected Dense layers with Dropout
- Output layer with Sigmoid activation

## Results
The model achieves good accuracy in detecting PTSD, as demonstrated by the classification report and accuracy score.

## Usage
To train the model and make predictions, run the provided script. The trained model will be saved as ptsd_detection_model.h5.

## Requirements
- numpy
- pandas
- tensorflow
- keras
- sklearn

## How to Run
Ensure you have the required libraries installed.
Prepare your dataset in the specified format.
Run the script to train the model and make predictions.

## Display Predictions
The script includes a function to evaluate the model on test data, displaying classification metrics such as precision, recall, and F1 score.

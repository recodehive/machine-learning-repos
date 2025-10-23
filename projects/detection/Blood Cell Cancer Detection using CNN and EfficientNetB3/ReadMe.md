
# Blood Cell Cancer Classification using CNN and EfficientNetB3

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Getting Started](#getting-started)
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
- [Future Work](#future-work)
- [Authors](#authors)
- [License](#license)

## Overview
This project aims to classify blood cell images to detect cancerous cells using deep learning techniques, specifically Convolutional Neural Networks (CNN) and EfficientNetB3 architecture. The goal is to develop a robust model that can accurately differentiate between cancerous and non-cancerous blood cells.

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/), containing images of various blood cell types. The dataset is organized into folders for each cell type.

## Installation
Ensure you have the following libraries installed:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- PIL

Install the required libraries using:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python pillow
```

## Getting Started
1. Clone the repository:
    ```bash
    git clone https://github.com/recodehive/machine-learning-repos.git
    ```
2. Navigate to the project directory:
    ```bash
    cd machine-learning-repos/Detection Models/Blood Cell Cancer Detection using CNN and EfficientNetB3
    ```
3. Ensure you have the required libraries installed as mentioned in the [Installation](#installation) section.

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
The project demonstrates the capability of CNN and EfficientNetB3 in classifying blood cell images with high accuracy. The final model achieved an accuracy of XX% on the validation set.

## Future Work
- Explore the use of other pre-trained models.
- Implement more advanced data augmentation techniques.
- Deploy the model as a web application for real-time predictions.

## Authors
- [Sanjay KV](https://github.com/sanjay-kv)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can copy and paste this improved version into your ReadMe.md file.

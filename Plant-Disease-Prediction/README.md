# Plant Disease Detection using CNN

## Introduction

This project aims to develop a convolutional neural network (CNN) model to accurately detect and classify diseases in plant leaves from images, helping farmers manage crop health more effectively. The CNN model is trained on a dataset containing images of healthy and diseased plant leaves across various categories of diseases.

## Dataset

The dataset used for training and testing the CNN model is the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset), which contains images of various plant diseases and healthy plant leaves. The dataset includes images of several plant species, such as tomatoes, potatoes, apples, grapes, and more.

## Model Architecture

The CNN model architecture used for this project consists of multiple convolutional layers followed by max-pooling layers, dropout layers for regularization, and fully connected layers. The final layer uses the softmax activation function to classify the input image into different disease categories.

## Technology Stack

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Directory Structure
plant-disease-detection/
├── data/ # Folder to store datasets
├── models/ # Folder to save trained models
├── notebooks/ # Jupyter notebooks for experiments
├── src/ # Source code for the project
│ ├── init.py
│ ├── data_preprocessing.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
├── README.md # Project description and instructions
├── requirements.txt # Dependencies
├── CONTRIBUTING.md # Guidelines for contributing
└── .gitignore # Files and directories to ignore


## How to Use

1. **Clone the Repository:**
   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection

2. **Install Dependencies:**
   pip install -r requirements.txt
3.**Preprocess Data:**
   python src/data_preprocessing.py
4.**Preprocess Data:**
   python src/data_preprocessing.py
5.**Evaluate the Model:**
   python src/evaluate.py


Contributing
Contributions are welcome! Please read the CONTRIBUTING.md file for details on how to contribute to this project.

License
This project is licensed under the MIT License. See the LICENSE file for details.


You can replace the placeholder `your-username` with your actual GitHub username. If you need any further assistance, feel free to ask!

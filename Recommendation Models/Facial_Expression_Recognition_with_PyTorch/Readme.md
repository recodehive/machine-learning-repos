# Facial Expression Recognition Using PyTorch

## Overview
This project develops a facial expression recognition system using PyTorch. The model classifies facial expressions into categories such as happy, sad, angry, surprised, etc., by using a Convolutional Neural Network (CNN). The model is trained and tested using a dataset from Kaggle that contains labeled images of facial expressions.

## Dataset
The dataset used for this project is publicly available on Kaggle:  
[Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

The dataset includes multiple categories of facial expressions, which are used to train the CNN for classification.

## Setup

### Prerequisites
Before running the project, ensure you have the following libraries installed:

1. **Albumentations**: This library is used for data augmentation.
   ```bash
   pip install -U git+https://github.com/albumentations-team/albumentations
   ```

2. **TIMM (PyTorch Image Models)**: This is a collection of pre-trained models and utilities for computer vision tasks.
   ```bash
   pip install timm
   ```

3. **OpenCV**: Ensure you have the latest version, especially with the OpenCV-contrib modules.
   ```bash
   pip install --upgrade opencv-contrib-python
   ```

#### You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

### Running the Notebook
1. **Download the Dataset**: Download the dataset from Kaggle and place it in the appropriate directory. Make sure to update any file paths in the notebook accordingly.
   
2. **Run the Notebook**: Execute the cells in the Jupyter notebook (`Facial_Expression_Recognition_with_PyTorch.ipynb`) to train the model on the dataset and evaluate its performance.

## Model Architecture
The facial expression recognition system is built using a Convolutional Neural Network (CNN), a class of deep learning models commonly used for image-related tasks. The model architecture is designed to classify images of faces into distinct facial expression categories such as happy, sad, angry, surprised, etc.

#### Key Components:

- **Convolutional Layers**: Extract spatial features by applying filters to detect patterns (edges, corners, textures) essential for identifying facial expressions.
- **Pooling Layers**: Downsample feature maps to reduce dimensionality while retaining important information, enhancing efficiency and mitigating overfitting.
- **Fully Connected Layers**: Process extracted features for final classification into emotion categories.

- **Pre-trained Models from TIMM**: Utilizes pre-trained models like EfficientNet or ResNet from the TIMM (PyTorch Image Models) library to leverage learned knowledge from large datasets, improving training speed and accuracy.

- **Data Augmentation with Albumentations**: Enhances the dataset through transformations (e.g., rotations, flips, brightness adjustments), increasing variability and robustness against real-world conditions.

- **Loss Function and Optimizer**: Trained using categorical cross-entropy loss for multi-class classification. Common optimizers include Adam or SGD to minimize the loss.

- **Output Layer**: Features a softmax activation function, providing probabilities for each facial expression class. The class with the highest probability is selected as the predicted emotion.


## Conclusion
This project demonstrates how PyTorch and CNNs can be used to build a facial expression recognition system. By utilizing pre-trained models and augmenting the dataset, the system can accurately classify different facial expressions from images.


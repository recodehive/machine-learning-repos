

# Eye Disease Classification Using Deep Learning

## Overview

The Eye Disease Classification project aims to develop a robust model for the automated classification of retinal images into four distinct disease types: Glaucoma, Cataract, Normal, and Diabetic Retinopathy. Leveraging a diverse dataset sourced from reputable repositories, the project employs a Convolutional Neural Network (CNN) architecture, with a focus on utilizing the pre-trained VGG19 model for its image feature extraction capabilities.

## Dataset


The dataset used in this project consists of retinal images carefully curated from Kaggle, ensuring a balanced representation of four disease types. 
Dataset link: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification.

![countplot](https://github.com/somaiaahmed/ThereForYou/assets/52898207/0e801464-8732-4d2f-8bd8-aaf325e065da)

## Model Architecture

The chosen model architecture is based on the VGG19 CNN, known for its effectiveness in image classification tasks. Key details about the model's architecture, input size, convolutional layers, pooling, activation functions, and fully connected layers are provided in the Report.


## Data Processing

The dataset undergoes meticulous processing to prepare it for model training. This involves loading images, organizing them into a DataFrame, and creating data generators for training and validation. Data augmentation techniques, such as rotation and zooming, are applied to enhance the model's generalization capabilities. 

## Training

The training phase involves splitting the dataset into training and validation sets, employing data generators, and utilizing transfer learning with the pre-trained VGG19 model. The training process is monitored with checkpoints and early stopping mechanisms. 

![true  predict eye disease](https://github.com/somaiaahmed/ThereForYou/assets/52898207/ff85994c-a90d-4a83-a20d-50875f439220)


## Result 
![eye disease acc](https://github.com/somaiaahmed/Eye-diseases-classification/assets/52898207/c1759152-ee04-417d-b61c-3b2369a85eeb) 


## Evaluation
![Model Eval](https://github.com/somaiaahmed/Eye-diseases-classification/assets/52898207/cd10f3aa-88aa-43f4-bdef-d2f4ec1e883b)





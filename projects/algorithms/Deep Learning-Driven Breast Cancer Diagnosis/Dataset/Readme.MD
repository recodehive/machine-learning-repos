# Breast Cancer Detection using Deep Learning

## Overview
This project aims to develop a Breast Cancer Detection system using deep learning models, specifically focusing on ultrasound images. We implemented and compared the performance of three models: Multi-Layer Perceptron (MLP), VGG16, and ResNet50.

## Dataset
The dataset used for this project is the Breast Ultrasound Images Dataset from Kaggle, which contains ultrasound images labeled as normal, benign, and malignant.

Link to the dataset: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
# Breast Cancer Ultrasound Images Dataset



## Dataset Overview

- **Dataset Size**: Large
- **Format**: Images (Ultrasound)
- **Purpose**: Breast Cancer Detection using Deep Learning
- **Data Source**: Kaggle

## Instructions for Usage

1. **Download Dataset**: Click on the download link provided above to download the Breast Cancer Ultrasound Images Dataset from Kaggle.

2. **Dataset Structure**: Once downloaded, ensure the dataset is organized in the following structure within this directory:

```
Breast Cancer Ultrasound Images Dataset/
|- images/
  |- [Ultrasound Image Files]
|- README.md
```

3. **Data Usage**: The dataset is ready for use in your Deep Learning projects for breast cancer detection using ultrasound images. You can use this dataset for model training, validation, and testing purposes.

**Note**: Please cite the dataset source (Kaggle) appropriately if you use this dataset in your research or projects.


## Models
We employed three different deep learning models to classify the ultrasound images:
1. **Multi-Layer Perceptron (MLP)**
2. **VGG16**
3. **ResNet50**

## Run the notebook to get the model.keras file without which webapp won't work.

## Results
### Confusion Matrices
![Confusion Matrices](path_to_confusion_matrices_image)

### Training and Validation Loss & Accuracy

#### MLP
![MLP Loss and Accuracy](path_to_mlp_loss_accuracy_image)

#### VGG16
![VGG16 Loss and Accuracy](path_to_vgg16_loss_accuracy_image)

#### ResNet50
![ResNet50 Loss and Accuracy](path_to_resnet50_loss_accuracy_image)

### Classification Reports

#### MLP
```
              precision    recall  f1-score   support
           0       0.60      1.00      0.75       175
           1       0.00      0.00      0.00        82
           2       1.00      0.42      0.60        59
    accuracy                           0.63       316
   macro avg       0.53      0.47      0.45       316
weighted avg       0.52      0.63      0.53       316
```

#### VGG16
```
              precision    recall  f1-score   support
           0       0.87      0.93      0.90       175
           1       0.86      0.83      0.84        82
           2       0.92      0.78      0.84        59
    accuracy                           0.88       316
   macro avg       0.88      0.85      0.86       316
weighted avg       0.88      0.88      0.88       316
```

#### ResNet50
```
              precision    recall  f1-score   support
           0       0.59      0.99      0.74       175
           1       0.92      0.28      0.43        82
           2       0.00      0.00      0.00        59
    accuracy                           0.62       316
   macro avg       0.50      0.42      0.39       316
weighted avg       0.57      0.62      0.52       316
```

## Conclusion
Among the three models, VGG16 achieved the highest accuracy and performed the best in terms of precision, recall, and F1-score. Future work will involve further fine-tuning of the models and exploring additional data augmentation techniques to improve the classification performance.

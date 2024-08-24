# Pneumonia Detection from Chest X-Ray Images using Transfer Learning

Domain             : Computer Vision, Machine Learning

Sub-Domain         : Deep Learning, Image Recognition

Techniques         : Deep Convolutional Neural Network, ImageNet, Inception

Application        : Image Recognition, Image Classification, Medical Imaging


# Description

1. Detected Pneumonia from Chest X-Ray images using Custom Deep Convololutional Neural Network and by retraining pretrained model “InceptionV3” with 5856 images of X-ray (1.15GB).
  
2. For retraining removed output layers, freezed first few layers and fine-tuned model for two new label classes (Pneumonia and Normal).
   
3. With Custom Deep Convololutional Neural Network attained testing accuracy 89.53% and loss 0.41.

# Dataset Details

## Dataset Name            : Chest X-Ray Images (Pneumonia)
Number of Class         : 2

Number/Size of Images   : Total      : 5856 (1.15 Gigabyte (GB)) || 
                          Training   : 5216 (1.07 Gigabyte (GB)) ||
                          Validation : 320  (42.8 Megabyte (MB)) ||
                          Testing    : 320  (35.4 Megabyte (MB)) 

## Model Parameters
Machine Learning Library: Keras

Base Model              : InceptionV3 && Custom Deep Convolutional Neural Network

Optimizers              : Adam

Loss Function           : categorical_crossentropy

## For Custom Deep Convolutional Neural Network : 
Training Parameters

Batch Size              : 64

Number of Epochs        : 30

Training Time           : 2 Hours

## Output (Prediction/ Recognition / Classification Metrics)
Testing

Accuracy (F-1) Score    : 89.53%

Loss                    : 0.41


Precision               : 88.37%

Recall (Pneumonia)      : 95.48% (For positive class)

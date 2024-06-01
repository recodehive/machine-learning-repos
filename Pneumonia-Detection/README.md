
# Pneumonia Detection using Neural Networkd

This project contains code snippets for training and using a CNN model for image Pneumonia Detection task. 

## Introduction

Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.

## Model

### 1. Convolutional Neural Network (CNN)

- **Description:** CNNs are deep learning models particularly well-suited for image classification tasks. They consist of multiple layers of convolutional and pooling operations, followed by fully connected layers.
- **Training:** The provided script (`pneumonia-detection-using-cnn.ipynb`) demonstrates how to train a CNN model using TensorFlow/Keras. It includes data preprocessing steps, model architecture definition, compilation, training and saving the trained model.
- **Usage:** After training, the model can classify new images as "Real" or "Fake". The script prompts the user to upload an image for classification using a file dialog.

Accuracy: 0.921474337

## Requirements
- Python 3.11
- TensorFlow (for CNN)
- NumPy
- PIL (Python Imaging Library)
- tkinter (for file dialog)
- Google Colab (for running the code)

## Dataset
The dataset used for training these models can be found at [this Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/).

## Usage

1. **Training Models:**
   - Update the paths to your local dataset directory in the respective scripts.
   - Run each script to train the corresponding model. The trained models will be saved locally.

If you have any Queries or Suggestions, feel free to reach out to me.

[<img height="30" src="https://img.shields.io/badge/linkedin-blue.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />][LinkedIn]
[<img height="30" src="https://img.shields.io/badge/github-black.svg?&style=for-the-badge&logo=github&logoColor=white" />][Github]
<br />

[linkedin]: https://www.linkedin.com/in/arpitsengar/
[github]: https://github.com/arpy8

<h3 align="center">Show some &nbsp;❤️&nbsp; by starring this repo! </h3>
# Hand Digit Recognition

This project involves to correctly identify digits (0-9) from images of handwritten digits using Convolutional Neural Network. 

### Model Workflow:
1. Data Preparation: The MNIST dataset is commonly used for training and testing the model.

2. Model Building:
    - Input Layer: Accepts 28x28 pixel images.
    - Convolutional Layers: Extract features like edges and shapes.
    - Pooling Layers: Reduce spatial dimensions and extract dominant features.
    - Dense Layers: Fully connected layers for classification.
    - Output Layer: 10 units with softmax activation for digit classification.

3. Model Compilation:
    - Loss Function: Sparse categorical crossentropy to compare predicted and actual digit labels.
    - Optimizer: Adam optimizer to update model weights.
    - Metrics: Accuracy to evaluate model performance.

4. Model Training:
    - Fit the model on training data, specifying the number of epochs.
    - Optionally, use a validation split to monitor performance on a subset of the training data.

5. Model Evaluation:
    - Evaluate the model on the test data to measure its accuracy and performance.

6. Prediction:
    - Use the trained model to make predictions on new or unseen images.


### Usage

1. Clone the repository.
2. Install the required libraries (` matplotlib`, `tensorflow`, `keras`).
3. Run the provided Python script.
4. Follow the instructions to input the features and select the model for prediction.

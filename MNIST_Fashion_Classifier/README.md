# Fashion MNIST Classifier

This repository contains a CNN model trained on Fashion MNIST achieving 84.26% accuracy using Cross Entropy Loss and Optim optimizer.

**Installation**

1. Clone this repository:
   ```
   git clone <repository-url>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

**Model Structure**

1. **Convolutional Layer (conv1)**: Extracts initial image features.
2. **Pooling Layer (pool)**: Downsamples feature maps.
3. **Convolutional Layer (conv2)**: Learns complex patterns.
4. **Fully Connected Layer (fc1)**: Captures feature relationships.
5. **Fully Connected Layer (fc2)**: Enhances pattern understanding.
6. **Fully Connected Layer (fc3)**: Produces class probabilities.

**Usage**

Load trained weights, pass images through the network for predictions. Customize or fine-tune the model as needed.

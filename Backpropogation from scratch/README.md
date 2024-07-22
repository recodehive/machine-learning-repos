# Backpropagation in Neural Networks

## Overview

Backpropagation is a fundamental algorithm used for training artificial neural networks. It computes the gradient of the loss function with respect to each weight by the chain rule, efficiently propagating errors backward through the network. This allows for the adjustment of weights to minimize the loss function, ultimately improving the performance of the neural network.




# How Backpropagation Works

## Forward propogation

- Input Layer: The input data is fed into the network.
- Hidden Layers: Each layer performs computations using weights and biases to transform the input data.
- Output Layer: The final transformation produces the output, which is compared to the actual target to calculate the loss.

### Mathematical Formulation

$$
a_i^l = f\left(z_i^l\right) = f\left(\sum_j w_{ij}^l a_j^{l-1} + b_i^l\right)
$$


where f is the activation function, zᵢˡ is the net input of neuron i in layer l, wᵢⱼˡ is the connection weight between neuron j in layer l — 1 and neuron i in layer l, and bᵢˡ is the bias of neuron i in layer l.

## Backward propogation

- Compute Loss: Calculate the error (loss) using a loss function (e.g., Mean Squared Error, Cross-Entropy Loss).
- Error Propagation: Propagate the error backward through the network, layer by layer.
- Gradient Calculation: Compute the gradient of the loss with respect to each weight using the chain rule.
- Weight Update: Adjust the weights by subtracting the gradient multiplied by the learning rate.

### Mathematical Formulation

- The loss function measures how well the neural network's output matches the target values. Common loss functions include:
1) **Mean Squared Error (MSE):** 

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
2) **Cross-Entropy Loss:**

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$


- For each weight 𝑤 in the network, the gradient of the loss L with respect to w is computed as:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}

$$


- Weights are updated using the gradient descent algorithm:
$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$


## Running the Notebook
```bash
git clone https://github.com/recodehive/machine-learning-repos/Backpropogation from scratch.git
```

```bash
pip install -r requirements.txt
```








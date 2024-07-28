# Activation Functions in Deep Learning: LaTeX Equations and Python Implementation

## Overview

This project provides LaTeX equations, explanations, and Python implementations for various activation functions used in Artificial Neural Networks (ANN) and Deep Learning. Our goal is to offer clear, visually appealing mathematical representations and practical implementations of these functions for educational and reference purposes.

## Contents

1. [Introduction to Activation Functions](#introduction-to-activation-functions)
2. [Activation Functions](#activation-functions)
3. [Mathematical Equations](#mathematical-equations)
4. [Python Implementations](#python-implementations)
5. [Jupyter Notebook](#jupyter-notebook)
7. [Comparison of Activation Functions](#comparison-of-activation-functions)
8. [How to Use This Repository](#how-to-use-this-repository)


## Introduction to Activation Functions

Activation functions are crucial components in neural networks, introducing non-linearity to the model and allowing it to learn complex patterns. They determine the output of a neural network node, given an input or set of inputs.

## Activation Functions

This project covers the following activation functions:

### Non-Linear Activation Functions
Non-linear activation functions introduce non-linearity into the model, enabling the network to learn and represent complex patterns.

-  Essential for deep learning models as they introduce the non-linearity needed to capture complex patterns and relationships in the data.

- Here are some common non-linear activation functions:
1. Sigmoid
2. Hyperbolic Tangent (tanh)
3. Rectified Linear Unit (ReLU)

### Linear Activation Functions
A linear activation function is a function where the output is directly proportional to the input.

- **Linearity:** The function does not introduce any non-linearity. The output is just a scaled version of the input.
- **Derivative:** The derivative of the function is constant, which means it does not vary with the input.

- Here are some common linear activation functions:

1. Identity
2. Step Function

## Mathematical Equations

We provide LaTeX equations for each activation function. For example:

1. Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$
2. Hyperbolic Tangent: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
3. ReLU: $f(x) = \max(0, x)$
4. Linear : $f(x) = x$
5. Step :  

$$
f(x) = 
\begin{cases} 
0 & \text{if } x < \text{threshold} \\
1 & \text{if } x \geq \text{threshold}
\end{cases}
$$


## Python Implementations

Here are the Python implementations of the activation functions:

```python
import numpy as np

# Non-Linear activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def reLu(x):
    return np.maximum(x, 0)

# Linear activation functions 
def identity(x):
    return x

def step(x, thres):
    return np.where(x >= thres, 1, 0)
```


## How to Use This Repository

- Clone this repository to your local machine.

```bash
  git clone https://github.com/recodehive/machine-learning-repos/Activation function
```
- For Python implementations and visualizations:

1. Ensure you have Jupyter Notebook installed 

```bash
  pip install jupyter
```
2. Navigate to the project directory in your terminal.
3. Open activation_functions.ipynb.

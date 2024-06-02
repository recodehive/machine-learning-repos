# Optimizers in Machine Learning

Optimizers are algorithms or methods used to adjust the weights and biases of your model to minimize the loss function.
They play a crucial role in training your model effectively.

## Types of Optimizers:

1. Gradient Descent:
    - Gradient Descent is the most basic optimization algorithm.The idea is to update the model's parameters in the opposite direction of the gradient of the loss function with respect to the parameters. 
    - Suitable for convex problems.
    - The size of the steps taken to reach the minimum is determined by the learning rate.
    - It can be slower for larger dataset and may get stuck in local minima for non-convex problems.
    - Formula : `y_new = y_old - alpha*f'(x)` where y_new is the updated parameter vector and y_old is the old parametr vector alpha is the step size that represents how far to move against each gradient with each iteration.

2. Stochastic Gradient Descent(SGD):
    - Stochastic Gradient Descent (SGD) is a variant of the gradient descent optimization algorithm where we use only one training example to calculate the gradient and update the parameters., which can lead to faster but noisier updates.
    - At each iteration, a random training example is selected.Thus, SGD uses a higher number of iterations to reach the local minima.
    - Formula : `w = w - η*∇Q(w)` where w is the parametr vector , η is the learning rate and ∇Q(w) is the Gradient of the Loss Function 

3. Stochastic Gradient Descent With Momentum:
    - Momentum in SDG helps accelerate gradient vectors in the right directions, thus helps in faster convergence of the loss function.
    - Using a large momentum and learning rate to make the process even faster might result in poor accuracy and even more oscillations.
    - Formula : `v(t) = γ*v(t−1) + η*∇J(θ)` where v(t) is the velocity at iteration (t), γ is the momentum factor ,η is the learning rate and ∇J(θ) is the gradient of the loss function with respect to the parameters (θ).

4.  Mini-Batch Gradient Descent:
    - Mini-Batch Gradient Descent is a variat where instead of a single training example or the whole dataset, Uses a subset(mini-batch) of the dataset to compute the gradient.
    - Mini-batch gradient descent is ideal and provides a good balance between speed and accuracy but it requires tuning of batch size..

5. AdaGrad:
    - It uses different learning rates for each iteration the change in learning rate depends upon the difference in the parameters during training.
    - Parameters with frequently occurring gradients have smaller learning rates, and those with infrequent gradients have larger learning rates.
    - This can be useful for dealing with sparse data.

6. RMS(Root Mean Square)Prop:
    - RMSprop is an extension of Adagrad that deals with its main drawback: the learning rate can become infinitesimally small over time. RMSprop fixes this by using an exponentially decaying average of squared gradients to scale the learning rate.
    - It is Effective for non-convex problems.

7. AdaDelta:
    - AdaDelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
    - It combines the advantages of both RMSprop and AdaGrad and also deal with there significant drawbacks by keeping an exponentially decaying average of past gradients and past squared gradients.

8. Adam (Adaptive Moment Estimation):
    - Adam combines the ideas of momentum and RMSprop. It computes adaptive learning rates for each parameter and maintains running averages of both the gradients and the squared gradients. It is widely used due to its efficiency and effectiveness.
    - It is efficient for large datasets and well-suited for non-convex optimization.



 
 


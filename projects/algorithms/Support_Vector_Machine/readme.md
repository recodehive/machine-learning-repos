### Support Vector Machine

Support Vector Machine (SVM) is a powerful machine learning algorithm which is used for linear or nonlinear classification as well as for regression. Its primary objective is to find a hyperplane that best divides a dataset into classes.

1. Hyperplane : A hyperplane is a decision boundary that separates different classes in the feature space. In a 2D space, this is simply a line, while in 3D, it's a plane, and in higher dimensions, it becomes a hyperplane.
There are many hyperplanes that might classify the data. One reasonable choice as the best hyperplane is the one that represents the largest separation, or margin, between the two classes. 

2. Margin : The margin is the distance between the hyperplane and the nearest support vectors. SVM aims to maximize this margin, ensuring the best separation between classes.

3. Support Vectors : These are the points that are closest to the hyperplane and which affect the position of the hyperplane. A separating line will be defined with the help of these data points.

Types of Support Vector Machine(SVM) Algorithm:
1. Linear SVM : It uses a linear decision boundary to separate the data points of different classes.Linear SVMs are very suitable
when the data is linearly separable meaning there is a straight line in 2D or hyperplane in higher dimensions that can separate the classes.
The distance between the hyperplane and a data point "x" can be calculated using the formula −  
```
distance = (w . x + b) / ||w|| 
```
where "w" is the weight vector, "b" is the bias term, "||w||" is the Euclidean norm of the weight vector, and "x" is a data point. The weight vector "w" is perpendicular to the hyperplane and determines its orientation, while the bias term "b" determines its position.

2. **Non-Linear SVM**: Used when data is not linearly separable. Employs kernel functions to transform the input data into a higher-dimensional space where it becomes linearly separable.
- **Kernel**: A mathematical function used in SVM to map the original input data points into high-dimensional feature spaces, making it easier to find a hyperplane. Common kernel functions include:
  * Polynomial Kernel: Suitable for polynomially separable data.
  * Radial Basis Function (RBF) Kernel: Suitable for more complex, non-linear data.
  * Sigmoid Kernel: Sometimes used in neural networks.

## Example: Breast Cancer Classification

We will use the breast cancer dataset which contains data used to diagnose breast cancer. It includes various features extracted from images of breast cancer cell nuclei, aiming to classify tumors as either malignant or benign.

### Code Explanation

The provided code:
- Loads the Breast Cancer dataset
- Splits it into training and testing sets
- Creates an SVM classifier with a linear kernel
- Trains the classifier with the training data
- Makes predictions on the test set
- Evaluates the model's accuracy

### Running the Code

To run ```svm.py``` code, ensure you have the necessary libraries installed (`sklearn`).

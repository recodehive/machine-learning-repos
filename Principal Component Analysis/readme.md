## Principal Component Analysis
Principal Component Analysis is an unsupervised learning algorithm that is used for the dimensionality reduction in machine learning. It is a statistical process that converts the observations of correlated features into a set of linearly uncorrelated features with the help of orthogonal transformation. These new transformed features are called the Principal Components. It is one of the popular tools that is used for exploratory data analysis and predictive modeling. It is a technique to draw strong patterns from the given dataset by reducing the variances. PCA generally tries to find the lower-dimensional surface to project the high-dimensional data. PCA works by considering the variance of each attribute because the high attribute shows the good split between the classes, and hence it reduces the dimensionality. Some real-world applications of PCA are image processing, movie recommendation system, optimizing the power allocation in various communication channels. It is a feature extraction technique, so it contains the important variables and drops the least important variable.

### Some common terms used in PCA algorithm:
1. Dimensionality: It is the number of features or variables present in the given dataset. More easily, it is the number of columns present in the dataset.
2. Correlation: It signifies that how strongly two variables are related to each other. Such as if one changes, the other variable also gets changed. The correlation value ranges from -1 to +1. Here, -1 occurs if variables are inversely proportional to each other, and +1 indicates that variables are directly proportional to each other.
3. Orthogonal: It defines that variables are not correlated to each other, and hence the correlation between the pair of variables is zero.
4. Eigenvectors: If there is a square matrix M, and a non-zero vector v is given. Then v will be eigenvector if Av is the scalar multiple of v.
5. Covariance Matrix: A matrix containing the covariance between the pair of variables is called the Covariance Matrix.

### Explaination of the code
1. Standardize the Data:

Extract feature values from the dataset.
Normalize the features to ensure they have a mean of zero and a standard deviation of one.

2. Verify Normalization:

Check that the normalized data has a mean of approximately zero and a standard deviation of one.

3. Convert Data to Tabular Format:

Create descriptive column names for each feature.
Store the normalized data in a DataFrame.

4. Perform Principal Component Analysis (PCA):

Import the PCA module and specify the number of principal components (e.g., 2).
Fit the PCA model to the normalized data and transform it to get the principal components.

5. Create DataFrame for Principal Components:

Create a new DataFrame to store the principal component values for each sample.

6. Determine Explained Variance Ratio:

Calculate the explained variance ratio for each principal component to understand the amount of variance captured by them.

7. Visualize the Principal Components:

Set up the plotting parameters (e.g., figure size, axis labels, title).
Assign colors and labels to different classes (e.g., 'Benign' and 'Malignant').
Plot the samples on a scatter plot using the principal component values, differentiating the classes by color.

8. Display the Plot:

Add a legend to the plot to indicate which color represents which class.
Show the final visualization to understand the distribution of samples along the principal components.

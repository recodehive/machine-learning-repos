# Regression

Regression in machine learning refers to a type of supervised learning algorithm used to predict continuous values.
Regression helps us understand the relationship between independent variables (features) and dependent variables (target) and predict a continuous output based on the input data.

## Types of Regression

1. Linear Regression:
    - the simplest form of regression that assumes a linear relationship between the independent variables and the dependent variable.
    - the dependent variable is continuous, independent variable(s) can be continuous or discrete.
    - represented by an equation `y = b*x + a + e` where y is the dependent variable , x is the independent varaible ,a is intercept, b is slope of the line and e is error term.
    - Example : Predicting house prices based on features such as area, number of bedrooms, number of bathrooms, etc.

2. Polynomial Regression:
    - Polynomial regression is an extension of linear regression where the relationship between the independent and dependent variables is modeled as an nth degree polynomial.
    - represented by the equation `y = b1*x + b2*x^2 + ..... +bn*x^n + a + e `
    - Example : Predicting the height of a plant based on the age of the plant. The relationship may not be linear; it could be quadratic or cubic, requiring a polynomial regression model to capture it accurately.

3. Logistic Regression:
    - logistic regression is used for binary classification rather than regression. It models the probability that an instance belongs to a particular class.
    - The logistic regression model uses the logistic function g(z), where z is the linear combination of the input features and their corresponding coefficients: `z = a + b1*X1 + b2*X2 + ...... + bn*Xn`
    the logistics function g(z) is defined as `g(z) = 1/(1+e^(-z))` , g(z) is the predicted probability.
    - Logistic regression doesn’t require linear relationship between dependent and independent variables.  It can handle various types of relationships.
    - Example : Predicting whether an email is spam or not based on features like the sender's address, subject line, and content. Logistic regression can output the probability of an email being spam, enabling classification based on a threshold.

4. Decision Tree Regression:
    - Decision tree regression uses a decision tree to model the relationship between the independent variables and the target variable.
    - A decision tree is a tree-like structure where each internal node represents a "test" on an attribute (a feature), each branch represents the outcome of the test, and each leaf node represents a class label (in classification) or a numerical value (in regression).
    - Its leads to problem of overfitting and instability.
    - Example : Predicting the price of a used car based on features such as mileage, age, brand, etc. A decision tree can split the data into segments based on these features and predict the price within each segment.

5. Random Forest Regression:
    - Random forest regression is an ensemble learning method that combines multiple decision trees to improve predictive performance and reduce overfitting.
    - Example : Predicting the sales of a product based on various factors such as advertising expenditure, seasonality, competitor prices, etc. Random forest regression can capture complex relationships between these factors and the sales outcome.

6. Ridge Regression:
    - Ridge Regression is a technique used when the data suffers from multicollinearity (independent variables are highly correlated).
    - Ridge regression is a regularized version of linear regression that penalizes large coefficients to prevent overfitting.
    - Example : Predicting a person's salary based on various factors such as education, experience, location, etc. Ridge regression can help prevent overfitting if there are multicollinearity issues among the features.

7. Lasso Regression:
    - Similar to Ridge Regression, Lasso (Least Absolute Shrinkage and Selection Operator) also penalizes the absolute size of the regression coefficients.
    - Example : Identifying significant features in a dataset containing numerous variables. Lasso regression can be useful for feature selection by shrinking less important features' coefficients to zero.

8. ElasticNet Regression
    - ElasticNet is hybrid of Lasso and Ridge Regression techniques. 
    - Elastic-net is useful when there are multiple features which are correlated.



Implementation of linear regression is given in `linear_regression.ipynb`:
The code involves finding the coefficients (intercept and slope) that best fit the data according to the least squares criterion.



    
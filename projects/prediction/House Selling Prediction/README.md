### Boston Housing Price Prediction
This project builds a Linear Regression Model to predict house prices based on features such as crime rate, average number of rooms, distance to employment centers, property tax rate, and more. The model is trained using the Boston Housing Dataset.

### Table of Contents
- Introduction
- Dataset
- Requirements
- Conclusion

**Introduction**
The Boston Housing Price Prediction project uses a linear regression model to predict the median value of owner-occupied homes (in $1000s) in various suburbs of Boston. This prediction is based on a set of 13 features related to socioeconomic, geographical, and property-specific factors.

**Dataset**
The dataset used is the Boston Housing Dataset, which contains 506 samples and 14 attributes. The target variable is MEDV, the median house price. The features include:

- CRIM: Per capita crime rate by town.
- ZN: Proportion of residential land zoned for large lots.
- INDUS: Proportion of non-retail business acres per town.
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- NX: Nitric oxides concentration (parts per 10 million).
- RM: Average number of rooms per dwelling.
- AGE: Proportion of owner-occupied units built prior to 1940.
- DIS: Weighted distances to five Boston employment centers.
- RAD: Index of accessibility to radial highways.
- TAX: Full-value property tax rate per $10,000.
- PTRATIO: Pupil-teacher ratio by town.
- B: Proportion of African American population.
- LSTAT: Percentage of lower status of the population.
- MEDV: Median value of owner-occupied homes in $1000s (Target variable).

**Requirements**
Ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Conclusion
This model provides a basic linear regression implementation to predict house prices in Boston. While linear regression is useful for many tasks, there are likely more advanced techniques (e.g., decision trees, random forests) that could yield better results for this dataset. Future improvements may include using these models and performing cross-validation.
# **Data Cleaning Techniques** 

**Note** : Use the respective dataset used in the jupyter notebooks.

- **Complete Case Analysis** : In this strategy we drop the rows of a particular feature whose missing values is leass than 5%.
- **KNN-Imputer** : The KNN Imputer is a technique used in multivariate imputation to fill in missing values by considering the values of their k-nearest neighbors. This method leverages similarities between data points to impute missing values effectively, offering a versatile approach to handling missing data in a multivariate context.
- **Adding Missing Indicator** : In this strategy, we add a new column with respect to the column we which is having missing values. This column has boolean values (true, false) wherever there is a NaN value in the original column, then true value is assigned to this tuple of this column.
- **Mean Median Missing Data** : Simple Imputer is a practical solution for filling missing numerical values in a dataset. This method replaces missing entries with the mean, median, or a specified constant, providing a straightforward approach to address and mitigate the impact of missing numerical data in your dataset.
- **One Hot Encoding** : One-hot encoding in machine learning is the conversion of categorical information into a format that may be fed into machine learning algorithms to improve prediction accuracy. One-hot encoding is a common method for dealing with categorical data in machine learning.
- **Categorical Ordinal Encoding** : Ordinal encoding is a preprocessing technique used for converting categorical data into numeric values that preserve their inherent ordering. It is useful when working with machine learning models like neural networks that expect numerical input features.

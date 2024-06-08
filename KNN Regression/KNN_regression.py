import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load the dataset
file_path = '/kaggle/input/salary-dataset/Salary_dataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Step 2: Exploratory Data Analysis (EDA)
# Summary statistics of the dataset
print(df.describe())

# Scatter plot of YearsExperience vs Salary
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='YearsExperience', y='Salary')
plt.title('Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Step 3: Preprocess the data
X = df[['YearsExperience']]
y = df['Salary']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ensure n_neighbors does not exceed number of training samples
max_neighbors = min(20, len(X_train))
param_grid = {'n_neighbors': np.arange(1, max_neighbors + 1)}

# Step 4: Hyperparameter Tuning using GridSearchCV for KNN
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameter for n_neighbors
best_n_neighbors = grid_search.best_params_['n_neighbors']
print(f"Best n_neighbors: {best_n_neighbors}")

# Building and training the KNN regression model with the best parameter
knn_regressor = KNeighborsRegressor(n_neighbors=best_n_neighbors)
knn_regressor.fit(X_train, y_train)

# Evaluate the KNN model
y_pred_knn = knn_regressor.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = mse_knn ** 0.5
r2_knn = r2_score(y_test, y_pred_knn)

print(f"KNN - Mean Squared Error: {mse_knn}")
print(f"KNN - Root Mean Squared Error: {rmse_knn}")
print(f"KNN - R^2 Score: {r2_knn}")

# Cross-Validation for KNN
cv_scores_knn = cross_val_score(knn_regressor, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores_knn = (-cv_scores_knn) ** 0.5
print(f"KNN - Cross-Validated RMSE: {cv_rmse_scores_knn.mean()}")

# Step 5: Compare with other models

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_lr = linear_regressor.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - Mean Squared Error: {mse_lr}")
print(f"Linear Regression - Root Mean Squared Error: {rmse_lr}")
print(f"Linear Regression - R^2 Score: {r2_lr}")

# Decision Tree Regression
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)
y_pred_tree = tree_regressor.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = mse_tree ** 0.5
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Decision Tree - Mean Squared Error: {mse_tree}")
print(f"Decision Tree - Root Mean Squared Error: {rmse_tree}")
print(f"Decision Tree - R^2 Score: {r2_tree}")

# Step 6: Visualization

# Actual vs Predicted Salaries for KNN
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, edgecolor='k', alpha=0.75)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Salaries (KNN)')
plt.show()

# Error distribution for KNN
errors = y_test - y_pred_knn
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.xlabel('Prediction Error')
plt.title('Distribution of Prediction Errors (KNN)')
plt.show()

# Step 7: Save the trained KNN model
model_path = 'knn_regressor_model.pkl'
joblib.dump(knn_regressor, model_path)
print(f"KNN model saved to {model_path}")

# Save the scaler for later use
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

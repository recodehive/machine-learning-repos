# Iris Flower Species Prediction

This project involves the prediction of iris flower species using machine learning models. It includes data preprocessing, model training, evaluation, and prediction based on user input.

https://github.com/vidurAgg22/machine-learning-repos/assets/165144144/ac7057d6-a1cf-40d2-8371-1515ef6e7a7d

## Dataset

The Iris dataset is used, which contains features such as sepal length, sepal width, petal length, petal width, and the target variable, which is the class of iris species.

### Dataset Description

The dataset consists of 150 samples with 4 features and one target variable. It includes the following information:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Class (Iris Species)

### Exploratory Data Analysis (EDA)

Exploratory data analysis (EDA) is performed to gain insights into the dataset.

- First 5 rows of the dataset are displayed.
- Statistical summary of the dataset is provided.
- Class distribution is visualized to understand the distribution of different iris species.
- Pairplot and correlation heatmap are plotted to explore relationships between features.

## Model Training and Evaluation

Six machine learning models are trained and evaluated for iris flower species prediction:

1. Support Vector Machine (SVM)
2. Logistic Regression
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Decision Tree
6. Naive Bayes

For each model, the following steps are performed:

- Data is split into training and testing sets.
- Features are standardized.
- The model is trained using the training data.
- Predictions are made on the test data.
- Model performance is evaluated using classification report and confusion matrix.

## Predicting Iris Flower Species Based on User Input

A function `predict_iris_species` is defined to predict the iris flower species based on user input. The user provides values for sepal length, sepal width, petal length, and petal width. Then, the user selects a machine learning model, and the predicted species is displayed.

## Usage

1. Clone the repository.
2. Install the required libraries (`pandas`, `seaborn`, `matplotlib`, `scikit-learn`).
3. Run the provided Python script.
4. Follow the instructions to input the features and select the model for prediction.

## File Structure

- `iris_prediction.py`: Python script containing the code for data preprocessing, model training, and prediction.
- `README.md`: Markdown file describing the project and usage instructions.

## Author

Vidur Agarwal

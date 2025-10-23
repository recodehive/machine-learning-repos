# Salary-Prediction-Web-App

## Project Overview

This project aims to predict software developer salaries in 2023 based on various factors such as country, education level, and years of coding experience. Please note that as a beginner in the field of machine learning, the accuracy of the model may not be optimal. However, I am committed to continuously improving the model as I gain more knowledge and experience in the field. The dataset is available [here](https://insights.stackoverflow.com/survey).

## Features

- **Predict Page:** Allows users to input their country, education level, and years of coding experience to get an estimated salary.
- **Explore Page:** Provides visualizations of salary distributions based on different countries and years of experience.

## Technologies Used

- Python
- Streamlit: For building the interactive web application.
- Pandas: For data manipulation and preprocessing.
- Scikit-learn: For machine learning model training and evaluation.
- Matplotlib and Seaborn: For data visualization.

## Machine Learning Models

This project utilizes several machine learning models and techniques for predicting software developer salaries:

- **Decision Tree Regressor**: Used to build a decision tree-based model for regression tasks. It partitions the feature space into regions and makes predictions based on the average target value within each region.

- **Random Forest Regressor**: Employed as an ensemble learning technique to combine multiple decision trees and improve prediction accuracy. Each tree in the random forest is trained on a random subset of the data.

- **Linear Regression**: Applied for modeling the relationship between independent variables (features) and the dependent variable (salary) using a linear equation.

- **GridSearchCV with Decision Tree Regressor**: Utilized for hyperparameter tuning, GridSearchCV systematically searches for the optimal hyperparameters of the decision tree regressor, such as the maximum depth of the tree.


## Usage

1. Navigate to the Predict Page to input your details and get a salary estimate.
2. Explore the Explore Page to visualize salary distributions based on different factors.
3. Experiment with different inputs to understand how different factors affect salary predictions.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.








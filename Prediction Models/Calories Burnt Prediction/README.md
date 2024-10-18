## Calories Burn Prediction

### Project Overview
The "Calories Fat Burn" project aims to predict the number of calories burned based on various features such as user demographics, exercise duration, and physiological parameters. Utilizing the XGBoost regression algorithm, the model helps in understanding the relationship between exercise and calorie expenditure, enabling users to optimize their workouts for better fat burning.

### Table of Contents
- Installation
- Data Collection
- Data Processing
- Data Analysis
- Model Training
- Evaluation

### Installation
To run this project, you will need to install the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Data Collection
The data is collected from two CSV files:

1. calories.csv: Contains user IDs and calories burned.
2. exercise.csv: Contains user demographics and exercise details.

### Data Processing
The data is processed to create a combined DataFrame containing user demographics and calories burned. The categorical variable "Gender" is encoded into numerical values for model training.

### Data Analysis
Statistical analysis and visualization techniques are employed to understand the data distribution and correlations among features.

- Gender Distribution
- Age Distribution
- Correlation Heatmap

### Model Training
The XGBoost regressor is trained on the training dataset to predict calorie burn.

### Evaluation
The model's performance is evaluated using the Mean Absolute Error (MAE).

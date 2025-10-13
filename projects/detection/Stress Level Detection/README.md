# Stress Level Detection
- The Stress Level Detection project aims to predict stress levels based on various physiological and demographic features using machine learning algorithms. 
- The dataset used in this project contains information on individuals, including their age, heart rate, sleep hours, and gender.
- The goal is to classify individuals into different stress levels using models such as Logistic Regression, Random Forest, and Support Vector Machines (SVM).

## Prerequisites
- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn

To install: `pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn`

## Dataset
The dataset used for this project is a CSV file named `stress_data.csv`, which includes the following columns:
- `Gender`: Gender of the individual (categorical)
- `Age`: Age of the individual (numerical)
- `HeartRate`: Heart rate of the individual (numerical)
- `SleepHours`: Number of hours the individual sleeps (numerical)
- `StressLevel`: Level of stress (categorical, target variable)

# Usage
- Mount your Google Drive to access the dataset.
- Load the dataset using Pandas.
- Perform data cleaning, including handling missing values.
- Encode categorical variables and normalize numerical features.
- Split the data into training, validation, and test sets.
- Conduct exploratory data analysis (EDA) to visualize data distributions and correlations.
- Train models using Logistic Regression, Random Forest, and SVM.
- Evaluate the models using classification reports and accuracy scores.
- Use SMOTE to address class imbalance and re-evaluate the models.

# Results
- Logistic Regression, Random Forest, and SVM models were trained and evaluated.
- SMOTE was applied to balance the dataset, resulting in improved accuracy for the SVM model.

# Conclusion
This project demonstrates the process of detecting stress levels using machine learning techniques.
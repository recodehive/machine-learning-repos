# Exoplanet Detection Using Machine Learning Models

## Introduction
An exoplanet is a planet that orbits a star outside our solar system, and its presence can often be detected by analyzing the light fluctuations or dips observed when the exoplanet passes in front of its host star. 

This project aims to classify whether an object is an exoplanet or not based on its flux (luminosity) measurements using several classification models.

## Prerequisites
- Python 3.x
- ``pandas``
- ``numpy``
- ``matplotlib``
- ``seaborn``
- ``scikit-learn``
- ``xgboost``
- ``imbalanced-learn``
- Jupyter Notebook (optional)

### Install Required Libraries
``pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn``

## Data Preprocessing
1. **Data Loading**: The flux data is read into a Pandas DataFrame and any missing values are handled.
2. **Outlier Removal**: Outliers in flux values exceeding a threshold of ``0.25e6`` are removed to avoid skewing model performance.
3. **Feature and Target Variables**:
    - **X**: Contains flux values (FLUX.1 to FLUX.3197).
    - **y**: Binary labels (``LABEL``), where ``1`` indicates "Not Exoplanet" and ``2`` indicates "Exoplanet."
4. **Balancing the Dataset**:
    - The dataset is imbalanced with more observations of non-exoplanet stars than exoplanet stars.
    - To address this, SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the class distribution.
5. **Data Splitting**: The data is split into training and testing sets using an 80-20 split.
6. **Feature Scaling**: ``StandardScaler`` is used to normalize the flux values for better performance of machine learning models.

## Exploratory Data Analysis
1. **Class Distribution**: Visualizes the number of exoplanet and non-exoplanet samples using bar plots.
2. **Flux Comparisons**: Plots the flux values of a representative exoplanet and non-exoplanet sample to observe differences in their luminosity curves.
3. **Boxplots**: Analyzes the distribution of flux values across different classes.

## Model Training
Six different machine learning models are trained and evaluated on the preprocessed dataset and each model is trained using the balanced and scaled dataset, and predictions are made on the test data.

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree Classifier
- XGBoost
- K-Nearest Neighbors (KNN)

## Model Evaluation
The models are evaluated using the following metrics:
1. **Accuracy Score**: Measures the proportion of correctly predicted instances.
2. **Confusion Matrix**: Provides insight into true positive, true negative, false positive, and false negative predictions.

## Results
| Model                  | Accuracy | Confusion Matrix         |
|------------------------|----------|--------------------------|
| Logistic Regression    | 0.831266 | [[979, 39], [301, 696]]  |
| Random Forest          | 0.556328 | [[1018, 0], [894, 103]]  |
| SVM                    | 0.599504 | [[1003, 15], [792, 205]] |
| Decision Tree          | 0.565757 | [[989, 29], [846, 151]]  |
| XGBoost                | 0.620347 | [[1018, 0], [765, 232]]  |
| K-Nearest Neighbors    | 0.933002 | [[883, 135], [0, 997]]   |


- The **K-Nearest Neighbors (KNN) model** achieved the highest accuracy of 93% making it the best-performing model for this problem.

## Conclusion
- This project effectively demonstrates the application of various machine learning techniques to classify stars as exoplanets or non-exoplanets based on their flux data. 
- The K-Nearest Neighbors (KNN) model emerged as the most accurate classifier, achieving an impressive accuracy of 93%. 
- This high performance illustrates the potential of machine learning methods in exoplanet detection, enabling more efficient analysis of astronomical data.

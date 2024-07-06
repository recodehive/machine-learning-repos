#  Home Income Prediction using Gradient Boosting Machine (GBM)

This project predicts the household income of the families based on various input features like the age, educationa and occupation of main member in family, number o fdependents, number of working members in family, location ,marital status, working ownership and source of transport etc. The project is based on four machine leanrning models as GBM , XGBoost , Multiple_linear_regression , Random Forest and GBM is found to be most accurate.

The complete data analysis, processing, feature engineering and model evaluation nd training is performed under notebook.

## Data Set

The below csv dataset from kaggle is used as reference which contains nearly 10000+ rows (Home_income_dataset.csv) on which processing is performed to obtained a  processed data , all this processing is performed in  notebook (Home_income_prediction_notebook.ipynb) file.

The dataset link is are as follows :-
https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis

on this dataset, below processing are performed :
1) featue scaling and column reinitialization
2) errors and outliers removal using box plot
3) remove na,missing values , regularization etc
4) Drop duplicates , normalization , column dropping

(all this works ar depicted in Home_income_prediction_noteboook ipynb file)

The model is trained on processed data after data processing and feature engineering  and all works associated with it are depicted in Home_income_prediction_notebook.ipynb file.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to feature engineering model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**

2. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie charts
    2) violen plot using matplotlib and plotly
    3) box plot of numerical features
    4) count plot 
    5) bar charts
    6) Regression graphs of actual vs predicted for all models
    (refer images folder for this images and graph observation)


4. **Model Training and evaluation**: 
     The four machine learning model multiple linear regression ,XgBoost ,GBM, Random Forest machine are selected for model training over the inputed processed data:

     model is then loaded into application after installing and using joblib library and ui is created with the hep of html , css and js. you can find html files under templates directory and css files under static directory.

5. **Inference**: 
      Deployed the model with the help html,css and js web application to predict the Home income based on various input features.


## Libraries Used

1. **Joblib**: For downloading the KNN model
2. **Scikit learn**: For machine learning processing  and operations
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For Data manipulation.
5. **NumPy**: For efficient numerical operations.
6. **Seaborn** : for advanced data visualizations
7. **plotly** : for 3D data visualizations .
8. **requests** : requests for creating Htttp requests

## How to Use

1. **Clone the Repository**: 
    ```sh
    git clone url_to_this_repository
    ```

2. **Install Dependencies**: 
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Model**: 
    ```python
    Python run app.py
    ```

4. **View Results**:  The results such as predicted income is displayed based on various input parameters , ownership features like property and home ownership ,dependents ,working members of family ,age,occupation etc with the help of application based on html, css and js and various machine learning models. you may refer webcam.mp4 for output video display.

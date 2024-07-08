#  Sleep Disorder Prediction using Machine learning

This project predicts whether a person is suffering from the sleep disorder or not and if yes then which disorder like sleep apnea or insomnia based on the kaggle dataset. The user enters various inputs like age , Occupation , Gender , Sleep Duration , Quality of sleep , Pyhsical activity level , BMI , Blood pressure , Heart rate etc . It applies four ddifferent machine learning models like XGBoost , GBM , logistic_regression and Random forest pon data and after cross valdiation over 10 folds  80 % accurated XGBoost is applied overe the data.

The entire reference and setup instructions are there in this readem.md file.

The application is applied using streamlit gui package in python.

## Data Set

The below csv dataset from kaggle is used as reference which contains nearly 500+ rows(sleep_health_and_lifestyle_dataset..csv) on which porcessing is performed to obtained a  processed data processed_data_car.csv , all this processing is performed in first notebook (sleep_disorder_detection.ipynb) file.

The dataset link is are as follows :-
https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

on this dataset, below processing are performed :
1) featue scaling and column reinitialization
2) errors and outliers removal
3) remove na,missing values , regularization etc
4) Normalization
5) Correlation tests
(all this works ar depicted in sleep_disorder_detection.ipynb file)

The model is trained on sleep_health_and_lifestyle_dataset.csv file and all works associated with it are depicted in sleep_disorder_detection.ipynb file.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**: 
2. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie charts
    2) violin plots
    3) box plots
    4) count plots 
    5) heatmap or confusion matrix for four different models of machine learning
    6) model comparison graphs
    7) Roc curve
    8) Sunburst Graph
    9) Histogram
    (refer images folder for this images and graph observation)


4. **Model Training and evaluation**: 
     The four machine learning model random forest ,XGBoost ,logistic regression, gradient boosting machine are selected for model training over the inputed processed data:
     random forest accuracy : 84 %
     GBM accuracy : 91 %
     XGBoost accuracy : 91 %
     logistic regression accuracy : 81 %

     The 10 fold cross validation is then performed on GBM and XGBoost model to obtained a final average cross validated accuracy of 80 % with XGBoost model.

     This XGBoost model is then loaded into streamlit application after installing and using joblib library.

5. **Inference**: 
      Deployed the model with the help streamlit web application to detect the sleep disorder ( sleep apnea / insomnia / No disorder (healthy))

## Libraries Used

1. **Joblib**: For downloading the random forest model
2. **Scikit learn**: For machine learning processing  and operations
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For image manipulation.
5. **NumPy**: For efficient numerical operations.
6. **Seaborn** : for advanced data visualizations
7. **plotly** : for 3D data visualizations .
8. **Streamlit** : for creating gui of the web application.


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
    streamlit run app.py
    ```

4. **View Results**: The script will allow you to predict whether the person is suffering from which sleep disorder or not.
#  Lung Cancer Prediction Using Machine Learning

This project classifies if the person has lung cancer or not based on various input features from the kaggle dataset like chronic disorders ,anxiety , chest pain ,smoking, drinking alcohol , age , gender , peer pressure and influence ,likeliness of yellow fingers, fatigue etc. The four different machine learning models like GBM , XGBOOST , Random Forest and Naive Bayes classifier are used for this purpose and final model is fitted with the help og GBM. The model is deployed with the help of UI application created using Streamlit package in python.

I have also performeed a dep data analysis , EDA , feature engineerin , training and evaluation under ipynb notebook file. Please follow instructions or setup guidelines to setup this project on your local server under this complete readme.md reference file.

## Data Set

The below csv dataset from kaggle is used as reference which contains nearly 3000+ rows (Lung_cancer.csv) on which porcessing is performed to obtained a  processed data , all this processing is performed in first notebook (Lung_Cancer_Prediction.ipynb) file.

The dataset link is are as follows :-
https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset

on this dataset, below processing are performed :
1) featue scaling and column reinitialization
2) errors and outliers removal using box plot
3) remove na,missing values , regularization etc
4) Drop duplicates , normalization , column dropping, correlation test

(all this works ar depicted in Lung_Cancer_Prediction.ipynb file)

The model is trained on processed data after data processing and feature engineering  and all works associated with it are depicted in Lung_Cancer_Prediction.ipynb file.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to feature engineering model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**

2. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie charts
    2) violen plots
    3) box plot of numerical features
    4) count plot 
    5) heatmap or confusion matrix for four different models of machine learning
    6) correlation heatmap
    7) histogram
    (refer images folder for this images and graph observation)


4. **Model Training and evaluation**: 
     The four machine learning model random forest ,XgBoost ,Naive Bayes Classifier, gradient boosting machine are selected for model training over the inputed processed data:
     random forest accuracy : 51 %
     GBM accuracy : 56 %
     XGBOOST accuracy : 53 %
     KNN accuracy : 52 %

5. **Inference**: 
      Deployed the model with the help streamlit web application to classify and predict the occurence of lung cancer.

## Libraries Used

1. **Joblib**: For downloading the KNN model
2. **Scikit learn**: For machine learning processing  and operations
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For Data manipulation.
5. **NumPy**: For efficient numerical operations.
6. **Seaborn** : for advanced data visualizations
7. **plotly** : for 3D data visualizations .
8. **Streamlit** : for creating gui of the web application.
9. **requests** : requests for creating Htttp requests

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

4. **View Results**:  The script and project will allow you to predict if the person is suffering from lung cancer or not based on various input features like age, chest pain, diffuculty in breathing , squeezing, chronic disease , oeer pressure ,smoking, yellow fingers , drinking alcohol, anxiety ,fatigue etc using different machine learning models deploayed on web application using streamlit package in python.
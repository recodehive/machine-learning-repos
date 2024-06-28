#  Asthma Disease Prediction using Machine learning

This project predicts whether the person is suffering from the respiratory disease of Asthma or not. The porject classifies the person into three main categories as suffering from mild asthma, suffering from Moderate or high asthma  and Not suffering from asthma on streamlit based application.

The application takes the input like the person has various disorders like tiredness,running nose,nsaql congestion, difficulty in breathing or not, age group , gender and predict the likeliness of the disease.

## Data Set

The below csv dataset from kaggle is used as reference which contains nearly 30000+ rows on which porcessing is performed to obtained a 3000 row processed csv data asthma_detection.csv.

The dataset link is are as follows :-
https://www.kaggle.com/datasets/deepayanthakur/asthma-disease-prediction

on this dataset, below porcessing are performed :
1) recreation of new asthma output column
2) SMOTE (synthetic minority oversmapling technique) to manage class imbalance
3) feature scaling and feature engineering

and finally the processed data asthma_detection.csv is obtained which is used to train the model.

The entire work of model training is depicited in Asthma_detection_ML.ipynb file. kindly refer it along with final dataset asthma_detection.csv.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**: 
      Following Data preprocessing and feature engineering steps are performed :
      1. removal of missing values, duplicates ,oversampling
      2. reverse encoding and label encoding
      3. correlation test and matrices
      4. Outlier detection and removal

3. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie chart of old age poeple suffering  and not suffering from asthma
    2) Histogram of tiredness vs asthma category
    3) violen category plot for different disorder likeliness with asthma
    4) count plot of all classes to detect class imbalance
    5) Box plot for outlier detection

    (refer output folder for this images and graph observation as well as wep application output that is created using streamlit)
    along with these in model training and evaluation below graphs are plotted :
    1) confusion matrix and classification report for random forest model
    2) confusion matrix and classification report for SVM model
    3) comparison charts for svm and random forest model

4. **Model Training and evaluation**: 
     The two machine learning model random forest and support vector machine are selected for model training over the inputed processed data:
     random forest accuracy : 85 %
     support vector machine accuracy : 84 %

     The 10 fold cross validation is then performed on random forest model to obtained a final average cross validated accuracy of 84 % with 2% of deviation.

     this random forest model is then loaded into streamlit application after installign using joblib library.

5. **Inference**: 
      Deployed the model with the help streamlit web application to detect asthma from input features. 


## Libraries Used

1. **Joblib**: For downloading the random forest model
2. **Sckiti learn**: For machine learning processing  and operations
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
    streamlit run main.py
    ```

4. **View Results**: The script will allow you to predict whether the person is suffering from asthma or not based on input features.


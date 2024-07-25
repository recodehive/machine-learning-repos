#  Hair Fall Detection using Machine learning

This project involves developing a Streamlit web application for predicting hair loss using both input-based and image-based approaches. The input-based prediction utilizes a pre-trained Naive Bayes model to predict hair loss based on user-provided information such as genetics, hormonal changes, medical conditions, medications, nutritional deficiencies, stress levels, age, poor hair care habits, environmental factors, smoking, and weight loss. The image-based prediction employs a pre-trained CNN model to classify uploaded images as either "Hair" or "No Hair" by resizing and preprocessing the image before making predictions. The app is designed with an intuitive and user-friendly interface, featuring custom CSS for enhanced styling and a sidebar for navigation. Users can either fill out a form with their personal and health details or upload an image to receive a prediction about their hair loss status. The application also maps the model's numerical predictions back to human-readable labels and provides appropriate success or error messages based on the results.


## Data Set

The below csv dataset from kaggle is used as reference for csv model :

https://www.kaggle.com/datasets/amitvkulkarni/hair-health


The below image dataset from kaggle is used as reference for image model :

https://www.kaggle.com/datasets/adnanzaidi/hairnohair1


## Notebboks

hairfall_detection_csv.ipynb (csv model)
hairfall_detection_images.ipynb (images model)


## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**:

2. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie chart 
    2) violin plot 
    3) box plot of numerical features
    4) count plot 
    5) heatmap or confusion matrix for four different models of machine learning
    6) model comparison graphs
    7) line plots
    8) roc curve
    9) bar graph
    10) histogram
    11) correlation matirx
    (refer images folder for this images and graph observation)


4. **Model Training and evaluation**: 
    
    ### CSV Model :
     The five machine learning model random forest ,xgboost ,KNN, gradient boosting machine, Naive Bayes are selected for model training over the inputed processed data.
   
     The naive bayes machine model is then loaded into streamlit application after installing and using joblib library and performing 10 fold cross validation over it.

     ### image classifcation model :
     CNN model of deep learning is used for image classifcation which goes through various steps of data analysis, processing, feature engineering , scaling and data augmentation on batch size of 32.

5. **Inference**: 
      Deployed the model with the help streamlit web application to predict the hairfall with the help of input fields as well as image.


## Libraries Used

1. **Joblib**: For downloading the random forest model
2. **Scikit learn**: For machine learning processing  and operations
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For image manipulation.
5. **NumPy**: For efficient numerical operations.
6. **Seaborn** : for advanced data visualizations
7. **plotly** : for 3D data visualizations .
8. **Streamlit** : for creating gui of the web application.
9. **Tensorflow & Keras**: For image processing and functionalities


## How to Use

1. **Clone the Repository**: 
    ```sh
    git clone url_to_this_repository
    ```

2. **Install Dependencies**: 
    ```sh
    pip install -r requirements.txt
    ```

3. **Dowload the hair_nohair_classifer.h5 model** :

    ```sh
    https://drive.google.com/file/d/1r9hYvDPjsUEWqDh_v-VA2QQkseVSBhSu/view?usp=sharing
    ```
    (keep in same directory as app.py)

3. **Run the Model**: 
    ```python
    streamlit run app.py
    ```

4. **View Results**: The script will allow you to predict hairfall from both textual field input as well as image input.


## Demo :

https://github.com/user-attachments/assets/e80ac781-2b5a-4702-8e98-0dd8cb3e37a4




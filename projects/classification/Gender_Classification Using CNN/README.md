#  Gender Classification using Convolutional neural networks (CNN)

This project classifies the given input  image as well as live video stream from web camera into male or female based on gender with net accuracy of 90 % of CNN neural netwok model. The web application in based on flask framework and html and css languages are used for frontend design.


## Data Set

The below dataset from kaggle is used to train CNN model which contains nearly 40000+ training images and near about 6000 validation or testing images of two genders male and female.

The dataset link is are as follows :-
https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset

The entire work of model training is depcited in Gender_Classification.ipynb file. kindly refer it after installing and loading dataset from the above link.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to model training, evaluation and deployment :-

1. **Data Collection**:
      Gathered a Dataset of gender Classification images from kaggle install it and load it in your notebook   directory.

2. **Data Preprocessing**: 
      Following Data preprocessing steps are performed :
      1. Image scaling and normalization
      2. Image standardization , brightness, contrast and rotation adjsutment
      3. Data Augmentation and height and width adjustment

3. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie chaart of male vs female images in train and test distribution
    2) Histogram of aspect ration distribution vs frequency
    3) Joint plot of height and width of training an testing set image distribution
    4) count plot of train and test set distribution
    (refer output folder for this images and graph observation)

4. **Model Training and evaluation**: 
     The CNN model is trained over entire dataset with 5 epochs having following  charavteristics criteria :
     batch size : 32
     image size : 150,150,3
     layers : 3
     activation layers : relu, sigmoid
     max poooling layer type : 2D
     optimizer : Adam's
     loss type : binary cross_entropy

5. **Inference**: 
      Deployed the model with the help of flask web application to detect gender from images and webcam. 


## Libraries Used

1. **TensorFlow & Keras**: For building and training the deep learning model.
2. **OpenCV**: For image processing and visualization.
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For image manipulation.
5. **NumPy**: For efficient numerical operations.


## How to Use

1. **Clone the Repository**: 
    ```sh
    git clone url_to_this_repository
    ```

2. **Install Dependencies**: 
    ```sh
    pip install -r requirements.txt
    ```
3. **Install model from provided drive link**:
    - link : https://drive.google.com/file/d/1IF1ocCf8TWr48NG-Uu2Km2uuKEhIL1m2/view?usp=sharing
    - install model (model.h5) from the link and keep it in same directory as main.py

3. **Run the Model**: 
    ```python
    python main.py
    ```

4. **View Results**: The script will allow you to predict gender from input image or webcam.


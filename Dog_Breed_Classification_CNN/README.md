#  Dog Breed Classification using Convolutional Neural Network (CNN)

This project classifies the given inuted dog image into different breed categories as well as give information about the breed like dog's category's scientific name , life span , height , weight and other breed information. The web application is deployed using streamlit application.

## Data Set

The below dataset from kaggle is used to train CNN model which contains nearly 1000+ training images and near about 200 validation or testing images of 10 Dog breed categories which are as follows :-

['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']

The dataset link is are as follows :-
https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-datasethttps://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset

The entire work of model training is depcited in Dog_Breed_classification.ipynb file. kindly refer it after installing and loading dataset from the above link.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to model training, evaluation and deployment :-

1. **Data Collection**:
      Gathered a Dataset of dog breed calssification images from kaggle install it and load it in your notebook   directory.

2. **Data Preprocessing**: 
      Following Data preprocessing steps are performed :
      1. Image scaling and normalization
      2. Image standardization , brightness, contrast and rotation adjsutment
      3. Data Augmentation and height and width adjustment

3. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) count plot of train and test distribution
    2) ROC curve of CNN model for all categories
    3) graph of accuracy vs no of epochs
    4) graph of loss vs no of epochs
    5) confusion matrix of CNN model
    (refer output folder for this images and graph observation)

4. **Model Training and evaluation**: 
     The CNN model is trained over entire dataset with 10 epochs having following  charavteristics criteria :
     batch size : 32
     image size : 150,150,3
     layers : 3
     activation layers : relu, sigmoid
     max poooling layer type : 2D
     optimizer : Adam's
     loss type : binary cross_entropy

     model accuracy : 97.91 %
     model loss : 0.05

5. **Inference**: 
      Deployed the model with the help of flask web application to classify the dog breed in one of seven categories along with breed information form DOG CEO API. 

The entire working of the streamlit based ui application is depicted in webcam.mp4 file.

## Libraries Used

1. **TensorFlow & Keras**: For building and training the deep learning model.
2. **OpenCV**: For image processing and visualization.
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For image manipulation.
5. **NumPy**: For efficient numerical operations.
6. **Streamlit**: For Building Web Application.

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
    - link : https://drive.google.com/file/d/1aKDvGmG3r60D7YWwiy7zfkRvtHdim9N4/view?usp=sharing
    - install model (dog_breed.h5) from the link and keep it in same directory as app.py

3. **Run the Model**: 
    ```python
     streamlit run app.py
    ```

4. **View Results**: The script will allow you to classify the breed of the given inputed dog image and give breed informatiom like height,width,lifespan , nature, behaviour,scientific name etc using Dog CEO API for clasified breed type.


# Music Genre Detection

This project involves the classification of music genres using the GTZAN dataset. Various machine learning algorithms and neural networks are employed to achieve the best accuracy.

## Dataset

The dataset used for this project is the [GTZAN Music Genre Classification Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). It contains audio tracks categorized into different genres.

## Project Overview

1. **Data Cleaning and Preprocessing**
2. **Data Visualization**
3. **Model Training and Evaluation**
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Decision Tree
    - Random Forest
    - CatBoost Classifier
    - XGBoost Classifier
4. **Neural Network Implementation**
5. **Model Comparison**
6. **Best Model Selection and Prediction**

## Steps

### 1. Data Cleaning and Preprocessing

The initial step involved loading the dataset and performing necessary cleaning. This includes handling missing values, encoding labels, and normalizing the data.

### 2. Data Visualization

Visualizations were created to understand the waveforms of each genre. This helped in gaining insights into the data distribution and characteristics of different genres.

### 3. Model Training and Evaluation

Several machine learning models were trained and evaluated using accuracy as the metric. The results are as follows:

- **Logistic Regression**: 52.33%
- **K-Nearest Neighbors (KNN)**: 70.67%
- **Decision Tree**: 62.00%

The comparison of these models was visualized in a graph for better understanding.

#### Advanced Models

Further, advanced models were applied to improve accuracy:

- **Random Forest Classifier**: 78.00%
- **CatBoost Classifier**: 83.33%
- **XGBoost Classifier**: 77.33%

A comparison graph of these advanced models was also created.

### 4. Neural Network Implementation

A neural network was trained for 100 epochs, achieving a test accuracy of 75%. Accuracy and error plots were generated to visualize the training process.

### 5. Model Comparison

All the models were compared based on their accuracies. CatBoost Classifier was found to be the best performing model with an accuracy of 83.33%.


## Results

The best performing model is the **CatBoost Classifier** with an accuracy of 83.33%.

## Files

- `main.ipynb`: Contains the code for data cleaning, preprocessing, visualization, model training, evaluation, and predictions.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/music-genre-detection.git
    cd music-genre-detection
    ```

2. Run the Jupyter notebook:

    ```bash
    jupyter notebook main.ipynb
    ```

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- CatBoost
- XGBoost
- TensorFlow / Keras (for neural networks)
- Jupyter Notebook

## Conclusion

This project demonstrates the application of various machine learning and neural network techniques for music genre classification. The CatBoost Classifier was the best performing model, achieving an accuracy of 83.33%.

## Acknowledgements

- The dataset used in this project is provided by [Kaggle](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification).

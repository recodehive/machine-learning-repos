# Fake News Detection Project

<img src="https://socialify.git.ci/kapilsinghnegi/Fake-News-Detection/image?description=1&font=Source%20Code%20Pro&forks=1&issues=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&pulls=1&stargazers=1&theme=Dark" alt="Fake-News-Detection" width="1280" height="320" />


The project aims to develop a machine-learning model capable of identifying and classifying any news article as fake or not. The distribution of fake news can potentially have highly adverse effects on people and culture. This project involves building and training a model to classify news as fake news or not using a diverse dataset of news articles. We have used four techniques to determine the results of the model.

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Gradient Boost Classifier**
4. **Random Forest Classifier**

## Project Overview

Fake news has become a significant issue in today's digital age, where information spreads rapidly through various online platforms. This project leverages machine learning algorithms to automatically determine the authenticity of news articles, providing a valuable tool to combat misinformation.

## Dataset

We have used a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:
- True: Genuine news articles
- False: Fake or fabricated news articles

## System Requirements 

Hardware :
1. 4GB RAM
2. i3 Processor
3. 500MB free space

Software :
1. Anaconda
2. Python

## Dependencies

Before running the code, make sure you have the following libraries and packages installed:

- Python 3
- Scikit-learn
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Regular Expression

You can install these dependencies using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
pip install seaborn 
pip install re 
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/kapilsinghnegi/Fake-News-Detection.git
```

2. Navigate to the project directory:

```bash
cd fake-news-detection
```

3. Execute the Jupyter Notebook or Python scripts associated with each classifier to train and test the models. For example:

```bash
python random_forest_classifier.py
```

4. The code will produce evaluation metrics and provide a prediction for whether the given news is true or false based on the trained model.

## Results

We evaluated each classifier's performance using metrics such as accuracy, precision, recall, and F1 score. The results are documented in the project files.

## Model Deployment

Once you are satisfied with the performance of a particular classifier, you can deploy it in a real-world application or integrate it into a larger system for automatic fake news detection.
---

## Project Screenshots

#### Not a Fake News
![Not a Fake News](https://github.com/kapilsinghnegi/Fake-News-Detection/assets/118688453/3d079c46-118a-4c53-a515-43b9146001c5)

#### Fake News
![Fake News](https://github.com/kapilsinghnegi/Fake-News-Detection/assets/118688453/2f5262f7-801d-4293-824c-13c29fb97fed)

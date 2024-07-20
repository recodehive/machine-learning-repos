# Kidney Stone Prediction
A small, hard deposit that forms in the kidneys and is often painful when passed.

Kidney stones are hard deposits of minerals and acid salts that stick together in concentrated urine. They can be painful when passing through the urinary tract, but usually don't cause permanent damage.

The most common symptom is severe pain, usually in the side of the abdomen, that's often associated with nausea.
Treatment includes pain relievers and drinking lots of water to help pass the stone. Medical procedures may be required to remove or break up larger stones.

Treatment includes pain relievers and drinking lots of water to help pass the stone. Medical procedures may be required to remove or break up larger stones.



## Dataset
The dataset which is used here, is collected from Kaggle website. Here is the link of the dataset : https://www.kaggle.com/utkarshxy/kidney-stone-data.

## Goal
The goal of this project is to create a prediction model which will predict the success rate of kidney stone operation based on the stone's size and type of treatment.
************************************************************

## What Have I done?
1. Importing all the required libraries. Check [`requirements.txt`]
2. Upload the dataset and the Jupyter Notebook file.
3. Data Cleaning
4. Data Visualization
5. Prediction Models
    - KNN Algorithm
    - Logistic Regression
    - Random Forest Classifier
    - Decision Tree Classifier
    - Support Vector Machine Classifier
    - AdaBoost Classifier
    - Gradient Boosting Classifier
    - Gaussian Naive Bayes Classifier
    - MLP Classifier
6. Model comparison
7. Conclusion

******************************************************************
## Libraries used
1. Numpy
2. Pandas
3. Matplotlib
4. Sklearn
5. Seaborn

**********************************

## Model Comparison
We have deployed nine machine learning algorithms and every algorithm is deployed successfully without any hesitation. We have checked the accuracy of the models based on the accuracy score of each of the models. Now let's take a look at the scores of each models.

|Name of the Model|Accuracy Score|
|:---:|:---:|
|Logistic Regression|82.86|
|Decision Tree Classifier|82.86|
|Random Forest Classifier|82.86|
|Naive Bayes Algorithm|82.86|
|KNN Algorithm|82.86|
|Support Vector Machine Algorithm|82.86|
|Gradient Boosting Algorithm|82.86|
|AdaBoosting Classifier|82.86|
|Artificial Neural Network|82.86|

***************************************
## Conclusion
* For this project we have deployed nine different algortihms and every algorithm provides more or less same accuracy score, which is 82.86.
* To predict the kidney stone using this dataset we can use any of the above mentioned algorithms or models and deploy the final model.
* Here data limitation comes into play and it restricts the model to be more accurate.
* To make the model more accurate we need more attributes in the dataset and also the number of data must be increased so that the model can be trained better.
* Apart from the data limitations created by the dataset, the models are successfully deployed, and are predicting the outcome accurate enough.
* To work with this dataset in my opinion, every one should go with simple logistic regression, as that would be enough!
******************************************

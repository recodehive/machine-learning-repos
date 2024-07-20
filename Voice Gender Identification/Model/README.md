# Voice Gender Identification
In the present scenario gender identification is based on voice signal of human being. Several Automatic speech recognition systems uses gender identification and has proved to be of great importance. In today's technology gender identification is for speaker's identity in advance security system. In the proposed work, gender identification is done from voice signal by extracting the characteristics such as pitch, energy and mfcc.



## Goal
The goal of this project is to make a model which will identify the voice of men and women depending on the input given.

## Dataset
The dataset is collected from the Kaggle website. Here is the link for the dataset : https://www.kaggle.com/primaryobjects/voicegender.
**********************
## What Have I done
1. Importing all the required libraries. Check [`requirements.txt`]
2. Upload the dataset and the Jupyter Notebook file.
3. Exploratory Data Analysis
    - Distribution of data plotting using distplot
    - Feature distribution using Boxplot
    - Data Pre-processing
4. Model Creation
    - Logistic Regression
    - KNN Algorithm
    - Random Forest Classification
    - Decision Tree Classification
    - Support Vector Machine
    - Gradient Boosting
5. Parameter Tuning using GridSearch CV
6. Conclusion

*********************************
## Libraries used
1. Numpy
2. Pandas
3. Matplotlib
4. Sklearn
5. Seaborn
6. os

***********************************
## Model Comparison
We have deployed six machine learning algorithms and every algorithm is deployed successfully without any hesitation. We have checked the accuracy of the models based on the accuracy score of each of the models. Now let's take a look at the scores of each models.

|Name of the Model|Accuracy Score|
|:---:|:---:|
|Logistic Regression|98.17|
|Decision Tree Classifier|94.51|
|Random Forest Classifier|97.26|
|Gradient Boosting|97.46|
|KNN Algorithm|98.17|
|Support Vector Machine Algorithm|99.10|
| Support Vector Machine with Grid search on 10-fold CV|99.50|

**********************************

## Conclusion

**Comparing all those scores scored by the machine learning algorithms, it is clear that Support Vector Machine with Grid search on 10-fold CV is having the upper hand in case of this dataset and after this, we can use KNN algorithm, which is also having good score as compared to the other deployed algorithms**

Best Fitted Models ranking - 
1. Support Vector Machine with Grid search on 10-fold CV
2. KNN Algorithm
3. Logistic Regression
4. Gradient Boosting
5. Random Forest Classifier
6. Decision Tree Classifier


# Customer Purchase Behaviour Prediction : [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset)

### Overview 

This model is trained upon the dataset that contains information on customer purchase behavior across various attributes.

### Libraries Used 

- pandas , numpy : To handle the data
- scikit-learn : For training the model , testing the accuracy , generating classification report
- matplotlib , seaborn : For data visualization


### What have I done : 

1. Downloaded the dataset using Kaggle API
2. Imported necessary libraries
3. Generating data summary
4. Splitting of the data into training data and testing data
5. Training models
6. Hyperparameter Tuning using RandomizedSearchCv


### Model Results 

1. Gradient Boosting Classifer :
    - Without Hyperparameter tuning : 94.34%
    -  With Hyperparamter tuning : 94.67%

2. Logistic Regression :
   - Without Hyperparameter tuning : 70%
   -  With Hyperparamter tuning : 84%

3. Decision Tree Classifier:
   - Without Hyperparameter tuning : 90.33%
   -   With Hyperparamter tuning : 83%

### Conclusion :

`GradientBoostingClassifier` performed well on the dataset with a accuracy of nearly 95%

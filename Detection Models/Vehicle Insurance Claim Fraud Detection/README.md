# Vehicle Insurance Claim Fraud Detection


## GOAL

  
Vehicle insurance fraud involves conspiring to make false or exaggerated claims involving property damage or personal injuries following an accident so, It will Detect fraud claims and will help Insurance Firms to verify them properly again.


## DATASET

  
https://www.kaggle.com/shivamb/vehicle-claim-fraud-detection  

## DESCRIPTION
  
Vehicle insurance fraud involves conspiring to make false or exaggerated claims involving property damage or personal injuries following an accident so, It will Detect fraud claims and will help Insurance Firms to verify them properly again.Starting with cleaning and EDA I'll be going with Classification & SGD Classifier and will try to finalize the one with the highest accuracy

## WORK DONE

* Analyzed the data, searched for missing values, seperated categorical and continous data.
* Removed unnecessary columns
* Converting some column data to Meaning full numerical data
* Grouping and Dealing with lots of Object Type of Data
* Build Relation Between PolicyType with other Two Column
* Recreated BasePolicy Column and compared it with PolicyType and which bought some insightful results
* Provided Desired Values to Data which was available in Range.
* Performed Exploratory Data Analysis
* Split the dataset into Train and Test data.
* Trained model with the following algorithms:
	* Support Vector Classifier
	* Naive Bayes Classifier
	* KNN Classifier
    * Decision Tree Classifier
    * Random Forest Classifier
    * XGBoost Classifier
    * Logistic Regression
* Also evaluated the performance of the model with highest accuracy.


## MODELS USED

1. **Support Vector Classifier** - Support vector machine is a representation of the training data as points in space separated into categories by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. It is effective in high dimensional spaces and uses a subset of training points in the decision function, so it is also memory efficient.
2. **Naive Bayes Classifier** - Naive Bayes algorithm based on Bayes’ theorem with the assumption of independence between every pair of features. This algorithm requires a small amount of training data to estimate the necessary parameters. Naive Bayes classifiers are extremely fast compared to more sophisticated methods.
3. **K-nearest neighbors Classifier** - It is a simple algorithm to understand and can be used for classification analysis. Classification is computed from a simple majority vote of the K nearest neighbours of each point. This algorithm is simple to implement, robust to noisy training data, and effective if training data is large.
4. **Decision Tree Classifier** - Given a data of attributes together with its classes, a decision tree produces a sequence of rules that can be used to classify the data. Decision Tree is simple to understand and visualise, requires little data preparation, and can handle both numerical and categorical data.
5. **Random Forest Classifier** - Random forest classifier fits a number of decision trees on various sub-samples of datasets and uses average to improve the predictive accuracy of the model and controls over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement.  It results in reduction in over-fitting and random forest classifier is more accurate than decision trees in most cases.
6. **XGBoost Classifier** - XGBoost is a popular gradient-boosting library for GPU training, distributed computing, and parallelization. It’s precise, it adapts well to all types of data and problems, it has excellent documentation, and overall it’s very easy to use. 
7. **Logistic Regression** - Logistic regression is a machine learning algorithm for classification. In this algorithm, the probabilities describing the possible outcomes of a single trial are modelled using a logistic function. It is most useful for understanding the influence of several independent variables on a single outcome variable.



  
## LIBRARIES NEEDED

* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Plotly

  

## ACCURACIES

| **Model** | Accuracy | 
| --- | --- |
|1. Support Vector|94.033722 % |
|2. Naive Bayes |90.920882 % |
|3. K Nearest Neighbours|94.033722 % |
|4. Decision Tree|89.775184 % |
|5. Random Forest |94.163424 % |
|6. XGBoost |93.623000 % |
|7. Logistic Regression | 94.033722 % | 



## CONCLUSION

K Nearest Neighbours had the highest accuracy out of all the others, followed by Support Vector and Random Forest. Through this project, I learned how to apply various classification algorithms. We were having huge big data of more than 10000+ records with more than 30 columns because of which this much accuracy is attained.

## CONTRIBUTED BY

- Created by [@Priyankesh](https://github.com/priyankeshh), GSSoC 2024

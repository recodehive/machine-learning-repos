## Introduction

The purpose of this document is to provide an overview of the approach used for
Predictive Analytics to determine the Remaining Useful Life for the Electric Vehicle
battery state of charge. We have implemented an IoT predictive analytics solution
leveraging AI & ML.

## Significance

There are multiple strategies for Predictive Analytics. For the
current use case , a Supervised Regression model was used to
predict remaining useful lifetime (RUL) for the battery State of
Charge of the Electric Vehicle

## Intended audience
Executives, business users, data scientists, and developers.
Depending on the audience, the presenter can show:
<ol>
<li> Business capability</li>
<li>A deep dive into the technical parts</li>
</ol>

## Google Cloud products used
Google Cloud Storage, Datalab,, Compute Engine instances

## Problem
<p>Apply a supervised regression model to predict remaining useful lifetime (RUL) for the battery State of Charge of the Electric Vehicles.</p>

## Solution
<p>During the development journey, though several advanced machine learning models
have been tried that resulted in various levels of forecast accuracy, the production-ready
model is finalised based on the calculated accuracy (root mean squared error).</p>

## The solution analysis
<p>This section describes a few interesting facts about Predictive Analytics for intermittent time series IoT telemetry data that has been adopted to build the solution.
</p>

<u>The following are the key considerations towards this project </u>:
  <ol>
    <li> The relative value of Electric Vehicle battery State of Charge have been considered as the target variable for prediction.</li>
    <li>Smoothing Discrete signals - The dataset consists of signals that are transmitted through GSM per unit time . Hence the signals showed a discrete behaviour on reception.
    The purpose of smoothing was to do outlier detection and imputation of values that are not in the range
The following approaches were considered for smoothing the discrete signals :
  <ul>
    <li> Butterworth Low Pass filter</li>
    <li> RC filter</li>
    <li> Exponential Moving Averages</li>
   </ul>
 Exponential Moving Averages was found to be a better approach for discrete signal
 smoothing. </li>

<li> Defining trips in the dataset - A trip in the dataset is defined as a series of observations that have uniform differences in the corresponding time . A trip ends when the subsequent timestamp exceeds the defined space (observations with 100 ms time interval have been defined within the scope of the trip)</li>
<li>Upon data exploration, it has been observed that the battery State of Charge is typically not replenished between subsequent trips in the dataset </li>
<li>Upon data exploration, it was observed that the time lag determined through autocorrelation was not consistent across the data sets; hence summary statistics like mean, range, and variance of the data available in the lag window cannot go as an input feature to the  Predictive model
Given that the appropriate time lag, as determined by autocorrelation, is not
consistent across (datasets / trips), the use of summary statistics of the data in
each lag window for prediction was determined inappropriate. For instance, the
autocorrelation of the variable EVVSP, which denotes vehicle speed, for different
trip numbers resulted in visual disparities. Below are the results of
autocorrelation plots via statsmodels.graphics.tsaplots.plot_acf used on the
EVVSP values for trips 8 and 35, respectively. </li>
<li>Upon data exploration, the following 12 signals were found to have outliers. Thus,
outlier imputation was performed for these 12 signals. </li>

</ol>

## The end to end ML lifecycle

### The Dataset

The dataset consists of historical time series data, and every event is labelled​.

To determine the State of Charge of the battery to predict the RUL, the most important
signals are IoT signals.

For the current circumstances, over ninety variables provided insights about the battery
degradation status thereby helping to determine the State of Charge of the battery.

#### Training Data Set
The train data consists of multiple comma separated values (CSV) files comprising
information from nine vehicles and three months of telemetry data. The training dataset
consists of the first  1400  trips, in random order. The trips are defined by the variable 'combined'.

The variable ‘combined’ is a derived variable and is derived from IMEI (a vehicle
number) for a particular date and variable that gives a unique trip number identified
exclusively for the chosen IMEI

The validation data comprises of 98 trips identified in the ‘combined’ column
The test data includes data from 373 trips.


**Target variable**

The relative value of State Of Charge (EVSMA_EWMA) = EVSMA_delta is considered as
target variable.
The EVSMA_delta ensures a more consistent behaviour in calculating the charge loss
in the subsequent observations and captures a more uniform behaviour for training the
model.

**Prediction process**

The machine learning model predicts the relative state of charge (EVSMA_delta) for
every observation in the test data. Trip wise summation of EVSMA_delta is calculated to
find the total energy loss observed in the trip.

The summation, which indicates the total energy loss, is subtracted from the first
absolute value of the state of charge. This leads to the extrapolation of the state of
charge at the conclusion of the trip (e.g., that in the last observation in the test data).

Below represents the dataset primarily used for training the model and the input data
received from Maruti:

1. **Multivariate Time Series Data** - There is more than one time dependent variable.
    Each variable depends not only on its past values but also on other variables.
    This dependency is used to predict the battery State of Charge at a given unit of
    time.


## Exploratory Data Analysis through Feature Selection

Why Feature Selection:

The Dataset used in the project consists of data of very high dimensionality. Typically predictive models built with high dimensionalit data are not performant because:

1. **Training time​** increases exponentially with the number of features.
2. Models have increasing risk of ​ **overfitting ​** with increasing number of features.

How to use Feature Selection?

When the dataset is not prohibitively large, in order to decide what is the best subset of
features, a strong strategy is to try all possible subsets of features in the dataset. In this
case, the dataset is prohibitively large, so it is a better approach to sample in the space
of feature subsets by various heuristics.

Generally, when choosing a machine learning algorithm for feature selection , there are
many optimisation methods that decides the feature importance.For example :
a. The feature importance in a tree based models are calculated based on ​ **Gini
Index​** and ​ **Entropy.​** Therefore​ **​** Information criterion and statistical significance
tests are used for Random Forests.
b. XGBoost uses gradient boosting to optimize creation of decision trees in the
ensemble
c. Mutual Information

The following approaches were tried for feature selection​:

a. Mutual Information :
**Pros**: Mutual Information captures any kind of relationship viz. Linear as well as
Non-linear between two variables

**Cons** :

1. Mutual_Info regression does not provide relative score of features with respect
    to one another.
2. This approach has high computational complexity when applied to large
    datasets. This led to discarding the algorithm in this case.


b. RandomForest

c. XGBoost Classifier

It was found that Random Forest was the best choice applied for this dataset. RandomForest also provides ability to Prune Features because of which model trains more quickly and uses less memory because the feature set is reduced. This holds good for the dataset used for this project because it is huge Compared to other algorithms Random Forests help to select variables that are statistically significant and can adjust the strictness of the algorithm by adjusting the p values that defaults to 0.01 and the ​maxRuns ​(maxRuns refer to the number of times the algorithm is run. The higher the ​maxRuns​ the more selective you can get in picking the variables.


## Machine Learning Model

The following ML approaches were tried for building the predictive analytics model for
remaining useful cycle of EV battery State of Charge :
**The following advanced models have been tried and tested in Univariate analysis :**
a. LSTM: The LSTM captures Time Series data by combining several Neural
Networks together. In LSTM instead of feeding the data at each individual time
step, the data is provided at all time steps within a window, or a context is
provided, to the neural network.

### LSTM model for multiple time series data of varying length

Data Preparation for LSTM
**Scenario 1​** : The data has​ a different time series for each trip
**Scenario  2** :The previous predictions need to be incorporated into the current time
feature space in order to make a current prediction

The training data consists of data shuffled for multiple trips where the sequence within
each trip remains unaltered.
In this exercise trip numbers less than 1600 is considered for Training data
Trip numbers greater than 1600 are for test dataset

LSTMs are sensitive to the scale of the input data, specifically when the sigmoid
(default) or tanh activation functions are used. ​Limit the range of input and output
values. Normalization brings the model data in a standard format that makes the
training faster and accurate.

Padding technique to fix the variable size input problem

The initial consideration towards this project was ​to train the model ​in batches for
which the sequence of time series trips need to have the same length.
In this approach of applying LSTM model with padding , the maximum and minimum
length of the trips were calculated .Padding sequences with zeroes were performed on
shorter trips so that the trips are of the same length and batch optimization was
attempted.

**Cons : ​** The cons of the padding approach was that
● First, it added to computational complexity as the trip length varied to a great
extent with respect to short and long trips
● Second, masking the target variable value seemed to be difficult in the test data
which altered the prediction results
● Third, the length of trip sequence in the test data was unpredictable which added
to the complexity in prediction results
● Fourth, If the length of the trip based test data exceeds the maximum trip length
in training data set (Also as defined in the padding sequence) , the result may
throw an error because of data dimension disparity

Model building with LSTM padding was dropped due to the consequences of going with
the approach as stated above.

Model building without padding the trip sequence
The approach used in the second approach of model building is sequential LSTM
dense model without padding the trip sequence

The trip sequence length was maintained as the original length , keeping the batch size
=1 and with undefined length of the time sequence .The time required to train the model
was taking considerable more time with batch size = 1


Subsequently when the batch size was increased to 1600 , the time required to train them model became optimal as depicted in the results below The sequence imposes an order on the observations that must be preserved when
training the models and making predictions.

Accuracy achieved for LSTM sequential dense without padding


### XGBoost for Time Series Forecasting as Supervised Learning

In this exercise the following exercise was considered for Training data set
‘Combined’

Following pre-processing activities were followed

1. Dropping the EVSMA_EWMA column as training will be performed against the
    EVSMA_delta value
2. Scale the EVSMA_delta value so that it’s values become more significant


**Runtime Parameters decided for XGBoost based on the results from Grid Search CV
for Hyperparameter tuning**

The Machine Learning model predicts the Delta State Of Charge (EVSMA_delta ) for
every observation in the test data. Trip wise summation of Delta State Of Charge is
carried to find the total energy loss incurred in the trip.

The summation which indicates the total energy loss is subtracted from the first
absolute value of State Of Charge.


**Final Note** : This leads to the prediction of the State Of Charge post the last observation
in the test data


###  Ensembling with XGBoost and LSTM

In the method of ensembling, ​the decisions from multiple models are combined to
improve the overall performance.
In this project, the​ average of predictions from XGBoost and LSTM were used to make
the final prediction

The model evaluation accuracy from the ensembled method can be found in the the
next section

### Conclusion for choosing the right model​ :

From the above results , it can be concluded that since the XGBoost model gave the
best accuracy , it will be the final choice to generate predictions on the test data


## Setting this up in a Google Cloud Platform project

The first time you set up the demo, allow from two to four hours. Later, the setup time
should drop to about 45 minutes.

### Prerequisites

Google Cloud Platform (GCP) projects form the basis for creating, enabling, and using
all GCP services including managing APIs, enabling billing, adding and removing
collaborators, and managing permissions for GCP resources.

#### Create GCP project

To create a new project:

```
● Go to the ​ Manage resources​ page in the GCP Console.
GO TO THE MANAGE RESOURCES PAGE
● On the ​ Select organization drop-down list at the top of the page, select the
organization in which you want to create a project. If you are a free trial user, skip
this step, as this list does not appear.
● Click ​ Create Project​.
● In the ​ New Project window that appears, enter a project name and select a billing
account as applicable.
● If you want to add the project to a folder, enter the folder name in the ​ Location
box.
● When you're finished entering new project details, click ​ Create​.
```

### model training

The details of the 2 models trained with the Datalab instance are specified in the
section 3.1.3 Machine Learning Model

The following VM instance was used in the project

n1-standard-32 (32 vCPUs, 120 GB memory)



For every upload option that is initiated , the example data file is stored in the Google Cloud
Storage bucket ....


### Output

All the output is stored on GCS, path :
The output contain prediction results, transformed data, model metrics and trained model.

<hr>

<b>Created by :</b>
<b><i> Prayag Thakur </i></b>
</br>
</br>
![Prayag](https://img.shields.io/badge/Prayag-%402019-orange.svg)
![status](https://img.shields.io/-pra/Status-up-green.svg)

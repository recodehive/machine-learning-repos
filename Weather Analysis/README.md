<h1>Weather Analysis</h1>

**GOAL**

To build a machine learning model for predicting the Average Rainfall per month for a given atmospheric conditions like temperature, humidity , dewpoint, pressure, windspeed, etc.

**DATASET**

[https://www.kaggle.com/datasets/mastmustu/weather-analysis]

**DESCRIPTION**

To analyze the dataset of Weather Analysis and build and train the model on the basis of different features and variables.

The datasets have a csv file with 3902 entries, 22 columns.

**Columns Description**:

- Date
- Average temperature (°F)
- Average humidity (%)
- Average dewpoint (°F)
- Average barometer (in)
- Average windspeed (mph)
- Average gustspeed (mph)
- Average direction (°deg)
- Rainfall for month (in)
- Rainfall for year (in)
- Maximum rain per minute
- Maximum temperature (°F)
- Minimum temperature (°F)
- Maximum humidity (%)
- Minimum humidity (%)
- Maximum pressure
- Minimum pressure
- Maximum windspeed (mph)
- Maximum gust speed (mph)
- Maximum heat index (°F)
- Month


### Visualization and EDA of different attributes:

<img alt="Distribution" src="./Images/distribution plot 1.png">

<img alt="Distribution" src="./Images/distribution plot 2.png">

<img alt="Regression" src="./Images/avg barometer vs rainfall per mnth.png">

<img alt="Regression" src="./Images/avg dewpoint vs rainfall per mnth.png">

<img alt="Regression" src="./Images/avg humidity vs rainfall per mnth.png">

<img alt="Regression" src="./Images/avg temp vs rainfall per mnth.png">

<img alt="Regression" src="./Images/avg windspeed vs rainfall per mnth.png">

<img alt="Regression" src="./Images/max temp vs rainfall per mnth.png">

<img alt="Regression" src="./Images/month vs rainfall per month.png">


**MODELS USED**

| Model                     | MSE_train | R2_train | MSE_test  | R2_test   |
|---------------------------|-----------|----------|-----------|-----------|
|Random Forest Regression	  | 0.0126    |	0.965291 | 0.082938	 | 0.773470  |
|XGBoost Regression	        | 0.0056    |	0.984504 | 0.089369	 | 0.755905  |
|Decision Tree	            | 0.58e-34  | 1.000000 | 0.144070	 | 0.606500  |
|Riddge Regression	        | 3.58e-34	| 1.000000 | 0.144070  | 0.606500  |
|Linear Regression	        | 0.274    	| 0.243614 | 0.281541  | 0.231021  |
|Elastic Net Regression	    | 2.94e-01	| 0.190594 | 0.302724	 | 0.173166  |
|Neural Network Regression	| 0.358     | 0.076272 | 0.405645	 |-0.107945  |


**WHAT I HAD DONE**

* Load the dataset which is CSV format.
* It has 3902 entries(Rows), 22 columns.
* Checked for missing values and cleaned the data accordingly.
* Analyzed the data, found insights and visualized them accordingly.
* Found detailed insights of different columns with target variable using plotting libraries.
* Train the datasets by different models and saves their accuracies into a dataframe.


**LIBRARIES NEEDED**

1. Pandas
2. Matplotlib
3. Sklearn
4. NumPy
5. XGBoost
6. Tensorflow
7. Keras
8. Sci-py
9. Seaborn



**CONCLUSION**

- Random Forest and XGBoost Regression models show promising performance with lower MSE and higher R-square values for both training set and dataset.
- Decision Tree Regression achieved perfect R-square on the training set but on the test set it's value is 0.6, indicating overfitting.
- Deep Neural Network (NN) has a high MSE and negative R-square on testing set, approximately zero on training set, suggesting poor performance on both training and test sets.


**AUTHOR**

- Created by [@Priyankesh](https://github.com/priyankeshh), GSSoC 2024
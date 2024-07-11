#  Health Insurance Price Prediction using Machine Learning

Project Summary :

Data Exploration and Preprocessing:

Dataset:  https://www.kaggle.com/datasets/annetxu/health-insurance-cost-prediction

Exploratory Data Analysis (EDA): 

Utilized Plotly and Seaborn for visualizations including pie charts, histograms, violin plots, and box plots to understand data distributions, correlations, and outliers.

Data Preprocessing:

Label Encoding: Converted categorical variables (sex, smoker, region) into numerical format using LabelEncoder from scikit-learn.
Handling Missing Values: Ensured data completeness by checking for and handling missing values appropriately.
Normalization: Used StandardScaler from scikit-learn for feature scaling where applicable.


Machine Learning Models:

Linear Regression (LR):

Trained a Linear Regression model to predict insurance charges based on features such as age, BMI, and others.
Evaluated using metrics like R-squared (accuracy), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).


Random Forest (RF):

Applied a Random Forest Regressor for prediction.
Evaluated performance metrics similar to LR.


XGBoost (XGB) and Gradient Boosting Machine (GBM):

Implemented XGBoost and GBM models for comparison.
Evaluated and compared their performance metrics with LR and RF.


Model Evaluation and Comparison:

Compared the performance of LR, RF, XGB, and GBM using metrics such as R-squared, MSE, RMSE, and MAPE.
Visualized the actual vs. predicted values using line plots and evaluated the accuracy across different models.


Deployment with Streamlit:

Developed a Streamlit web application for predicting insurance charges based on user inputs (age, sex, BMI, children, smoker, region).
Integrated the trained LR model and label encoders into the Streamlit app.
Provided a user-friendly interface where users can input their data and get the predicted insurance charge displayed in a visually appealing green container with bold text.


Future Directions:


Model Improvement: Fine-tuning models for better accuracy, exploring ensemble techniques or deep learning approaches if needed.
Feature Engineering: Further exploring feature interactions or transformations to enhance model performance.
User Experience: Improving the UI/UX of the Streamlit app, adding more features such as data visualization options and model selection.


Tools and Technologies Used:

Programming Languages: Python
Libraries and Frameworks: pandas, NumPy, scikit-learn, XGBoost, Plotly, Seaborn, Streamlit
Data Visualization: Plotly, Seaborn for interactive and insightful visualizations.
Machine Learning: Regression models (Linear Regression, Random Forest, XGBoost, GBM) for predictive analysis.
Web Application Development: Streamlit for creating interactive and user-friendly web applications.

Conclusion:

Your project revolves around leveraging machine learning techniques to predict insurance charges based on various customer attributes. The journey has included data exploration, preprocessing, model building, evaluation, and deployment using modern tools and frameworks. This structured approach ensures robust predictions and a seamless user experience through the Streamlit application.


## How to Use

1. **Clone the Repository**: 
    ```sh
    git clone url_to_this_repository
    ```

2. **Install Dependencies**: 
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Model**: 
    ```python
    streamlit run main.py
    ```

4. **View Results**: The script will allow you to predict the estimated cost of health insurance for a person














































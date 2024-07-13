#  Diamond price prediction prediction using Machine learning

 **Model Training and evaluation**: 
     The four machine learning model random forest ,decision trees linear regression , support vector machine are selected for model training over the inputed processed data:
     random forest accuracy : 98 %
     support vector machine accuracy : 87 %
     decision trees accuracy : 97 %
     linear regression accuracy : 90 %

     The 10 fold cross validation is then performed on  Random forest model to obtained a final average cross validated accuracy of 98.25 %.

## How to Use

1. **Clone the Repository**: 
    ```sh
    git clone url_to_this_repository
    ```

2. **Install Dependencies**: 
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the model in drive and Run the Model**: 
    
    drive link : https://drive.google.com/file/d/1OQ3WwkFdhlTmOptbfgudzCx6nS5QggQH/view?usp=sharing (keep it in same direcotry as app.py)

    ```python
    streamlit run app.py
    ```

4. **View Results**: The script will allow you to predict the estimated price of diamonds.
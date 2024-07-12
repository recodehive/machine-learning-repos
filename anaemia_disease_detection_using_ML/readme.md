# Anaemia Disease prediction using machine learning

The project encompassed several stages of development and analysis focused on predicting anemia through a Streamlit application. Initially, exploratory data analysis (EDA) was conducted using Matplotlib and Seaborn to visualize distributions, correlations, and feature characteristics such as pixel color percentages and hemoglobin levels. This facilitated a deeper understanding of the dataset's structure and potential predictive features.

Feature engineering involved preprocessing steps like label encoding categorical variables such as gender to numeric values for model compatibility. A Random Forest (RF) model was selected as the primary predictive tool due to its ability to handle non-linear relationships and provide robust predictions. Model training involved splitting the data into training and testing sets using Scikit-learn's train_test_split function.

Further model comparisons were made with Support Vector Machine (SVM), Gradient Boosting Machine (GBM), and Naive Bayes classifiers to evaluate their performance metrics such as accuracy, precision, recall, F1 score, and ROC curve analysis. SVMs were explored with polynomial kernels, while GBM utilized boosting techniques to improve prediction accuracy. Naive Bayes provided a baseline comparison against more complex models.

The Streamlit application integrated these models seamlessly, enabling users to input their data interactively and receive instant predictions regarding their anemia status. This user-friendly interface leveraged efficient model loading with joblib and ensured accurate predictions through robust preprocessing steps and model selection. Overall, the project highlighted the application of machine learning in healthcare diagnostics, promoting accessibility and ease of use for individuals concerned about their health status.


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
    streamlit run app.py
    ```
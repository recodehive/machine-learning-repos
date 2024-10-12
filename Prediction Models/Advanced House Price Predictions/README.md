
### Advanced House Price Prediction

This project utilizes the California housing dataset to predict housing prices based on various features using machine learning techniques. The primary goal is to explore the relationships between different features of the dataset and the median house value, then build a model that can accurately predict house prices.

### Table of Contents

- Dataset
- Installation
- Usage
- Data Exploration
- Model Training
- Results
- Contributions
- License


### Dataset:

The dataset used in this project is the California housing dataset, which includes the following features:

- **MedInc:** Median income in block group
- **HouseAge:** Median house age in the block
- **AveRooms:** Average number of rooms per household
- **AveBedrms:** Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup:** Average house occupancy
- **Latitude:** Geographical latitude
- **Longitude:** Geographical longitude
- **MedHouseVal:** Median house value (target variable)
- The dataset can be fetched directly using fetch_california_housing() from sklearn.datasets.

### **Installation**
To run this project, ensure you have Python installed on your machine. You will also need the following packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using pip:
  
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

Clone this repository:

  ```bash
  git clone https://github.com/yourusername/california-housing-price-prediction.git
  cd california-housing-price-prediction
 ```
Run the Jupyter Notebook or Python script:

```bash
jupyter notebook California_Housing_Price_Prediction.ipynb
```

### Data Exploration
The data exploration process includes:

- Displaying the first few rows of the dataset.
- Summary statistics of the features.
- Checking for missing values.
- Visualizing relationships between features using pair plots and scatter plots.
- Analyzing the distribution of the target variable (Median House Value).

### Model Training
The project utilizes a Random Forest Regressor to predict the median house value based on the input features. The workflow includes:
  
 1.  Data Preprocessing:
  
      - Splitting the dataset into training and testing sets.
      - Standardizing the features using StandardScaler.

  2. Model Training:
  
      - Training the Random Forest model with 100 estimators.

  3. Evaluation:

      - Evaluating the model's performance using Mean Squared Error (MSE) and R-squared metrics.


### Results
The model's performance metrics are as follows:

- Training MSE: 0.04
- Testing MSE: 0.26
- Training R²: 0.97
- Testing R²: 0.81

# Medical Appointment No-Show Prediction

## Description
This project aims to predict patient no-shows for medical appointments using machine learning. No-shows cause inefficiencies and lost resources in healthcare; predicting these helps optimize scheduling and patient outreach.

We explore several machine learning models, including Logistic Regression, Decision Trees, Random Forest, XGBoost, and Artificial Neural Networks (ANN). We perform data cleaning, feature engineering, model training, hyperparameter tuning, and evaluation.

## Project Structure
```
Medical_Appointment_No_Shows/
├── data/              # Dataset files (KaggleV2-May-2016.csv)
├── notebooks/         # Jupyter notebooks (main.ipynb)
├── src/              # Source code (Python modules)
├── models/           # Saved models
├── pictures/         # Visualizations and plots
├── research/         # Research materials
├── requirements.txt  # Dependencies
└── README.md        # Project documentation
```

## Dataset
**Source:** Kaggle Medical Appointment No-Shows Dataset

The dataset contains patient appointment information, including:
- **Patient demographics:** Age, scholarship status, health conditions (hypertension, diabetes, alcoholism, handicap)
- **Appointment information:** Scheduled day, appointment day, days difference
- **SMS reminders:** Whether patient received SMS reminder
- **Target variable:** No-show labels (0 = showed up, 1 = no-show)

**Key Challenges:**
- Imbalanced dataset with fewer no-shows than shows
- Time features needing careful processing (handling time components in dates)
- Feature selection and encoding to improve model learning

## Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Medical_Appointment_No_Shows

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. **Data Exploration:** Open `notebooks/main.ipynb` to explore the dataset
2. **Data Preprocessing:** The notebook includes data cleaning and feature engineering
3. **Model Training:** Train models using the provided code in the notebook
4. **Evaluation:** Review model performance metrics and visualizations

```python
# Example: Load and explore data
import pandas as pd
df = pd.read_csv('data/KaggleV2-May-2016.csv')
df.head()
```

## Model Details
### Algorithms Used:
- **Logistic Regression:** Baseline linear model
- **Decision Tree:** Non-linear decision boundary model
- **Random Forest:** Ensemble of decision trees with tuning
- **Support Vector Machine (SVM):** Kernel-based classifier
- **XGBoost:** Gradient boosting with class balancing

### Performance:
- **Baseline Accuracy:** ~77% (but poor recall on no-shows)
- **Tuned Random Forest F1 Score:** ~0.44
- **XGBoost Recall:** ~79% (improved no-show detection)

### Training Details:
1. Data Cleaning and Feature Engineering  
2. Exploratory Data Analysis with visualization (histograms, bar plots, correlation heatmaps)  
3. Baseline model comparisons (Logistic Regression, Decision Tree, Random Forest, SVM)  
4. Hyperparameter tuning using GridSearchCV with custom scoring (F1)  
5. Evaluation with classification metrics focusing on F1 score and recall  
6. Advanced models: XGBoost with balanced class weights

## Results
- Baseline models achieved up to ~77% accuracy but poor recall on no-shows
- Hyperparameter tuning improved F1 score significantly (~0.44), emphasizing recall
- XGBoost model achieved ~79% recall for no-show detection
- Final tuned models balanced recall and precision for effective no-show detection
- Feature importance analysis revealed key predictors (SMS received, days difference, age)

**Visualizations:** See `pictures/` folder for feature importance plots, confusion matrices, and other visualizations.

## Contributing
Contributions, suggestions, and feedback welcome! 

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License
MIT License

---


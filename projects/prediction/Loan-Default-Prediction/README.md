# Loan Default Prediction

## Description
A machine learning model to predict whether a loan applicant is likely to default on their loan. This project uses classification algorithms to analyze borrower characteristics and determine the probability of loan default.

## Project Structure
```
Loan-Default-Prediction/
├── data/                    # Dataset files
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
├── models/                  # Saved models
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## Dataset
The dataset contains loan application information including:
- Borrower demographics (age, income, employment status)
- Loan characteristics (amount, term, interest rate)
- Credit history (credit score, past defaults)
- Financial ratios (debt-to-income ratio)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.model import LoanDefaultPredictor

predictor = LoanDefaultPredictor()
predictor.load_model('models/loan_default_model.pkl')
prediction = predictor.predict(loan_data)
```

## Model Details
- **Algorithm**: Random Forest, XGBoost, Logistic Regression
- **Features**: 15+ engineered features
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.85 | 0.82 | 0.79 | 0.80 |
| XGBoost | 0.87 | 0.84 | 0.81 | 0.82 |
| Logistic Regression | 0.81 | 0.78 | 0.75 | 0.76 |

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License
MIT License

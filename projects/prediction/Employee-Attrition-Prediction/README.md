# Employee Attrition Prediction

## Description
A machine learning model to predict employee attrition (turnover) in organizations. This project helps HR departments identify employees who are likely to leave the company, enabling proactive retention strategies.

## Project Structure
```
Employee-Attrition-Prediction/
├── data/                    # Dataset files
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
├── models/                  # Saved models
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## Dataset
The dataset includes employee information such as:
- Demographics (age, gender, marital status, education)
- Job-related factors (department, job role, years at company)
- Compensation (salary, stock options, overtime)
- Work-life balance metrics
- Performance ratings and satisfaction scores

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.model import AttritionPredictor

predictor = AttritionPredictor()
predictor.load_model('models/attrition_model.pkl')
prediction = predictor.predict(employee_data)
```

## Model Details
- **Algorithm**: Random Forest, Gradient Boosting, Neural Network
- **Features**: 20+ engineered features including tenure, satisfaction index
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.88 | 0.85 | 0.82 | 0.83 |
| Gradient Boosting | 0.89 | 0.86 | 0.84 | 0.85 |
| Neural Network | 0.87 | 0.83 | 0.80 | 0.81 |

## Key Insights
- Overtime and work-life balance are top predictors of attrition
- Job satisfaction significantly impacts retention
- Employees with fewer years at company have higher attrition risk

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License
MIT License

# Energy Consumption Prediction

## Description
A machine learning model to predict energy consumption patterns for buildings, households, or industrial facilities. This project helps optimize energy usage and reduce costs through accurate forecasting.

## Project Structure
```
Energy-Consumption-Prediction/
├── data/                    # Dataset files
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
├── models/                  # Saved models
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## Dataset
The dataset includes energy consumption data with features such as:
- Temporal features (hour, day, month, season)
- Weather conditions (temperature, humidity, wind speed)
- Building characteristics (size, type, occupancy)
- Historical consumption patterns
- Holiday and weekend indicators

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.model import EnergyPredictor

predictor = EnergyPredictor()
predictor.load_model('models/energy_model.pkl')
prediction = predictor.predict(input_features)
```

## Model Details
- **Algorithm**: LSTM, XGBoost, Random Forest, Prophet
- **Features**: 25+ engineered features including lag variables
- **Metrics**: MAE, RMSE, MAPE, R-squared

## Results
| Model | MAE | RMSE | MAPE | R-squared |
|-------|-----|------|------|----------|
| LSTM | 45.2 | 62.3 | 8.5% | 0.92 |
| XGBoost | 48.1 | 65.7 | 9.1% | 0.90 |
| Random Forest | 51.3 | 68.9 | 9.8% | 0.88 |
| Prophet | 52.8 | 71.2 | 10.2% | 0.86 |

## Applications
- Smart grid optimization
- Building energy management
- Cost forecasting for utilities
- Demand response planning

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License
MIT License

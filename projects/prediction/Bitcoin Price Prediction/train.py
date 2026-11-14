import os
import joblib
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'coin_Bitcoin.csv')
    model_path = os.path.join(base_dir, 'bitcoin_model.pkl')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    bitcoin = pd.read_csv(dataset_path)
    for col in ['Name', 'SNo', 'Symbol', 'Date']:
        if col in bitcoin.columns:
            bitcoin = bitcoin.drop(columns=[col])

    feature_cols = ['High', 'Low', 'Open', 'Close', 'Volume']
    X = bitcoin[feature_cols]
    y = bitcoin['Marketcap']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Lasso(random_state=42)
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"Model trained. Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    joblib.dump(model, model_path)
    print(f"Artifact saved to {model_path}")


if __name__ == '__main__':
    train_and_export()

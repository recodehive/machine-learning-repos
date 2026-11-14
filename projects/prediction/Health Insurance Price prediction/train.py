import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'insurance.csv')
    model_path = os.path.join(base_dir, 'linear_regression_model.pkl')
    encoders_path = os.path.join(base_dir, 'label_encoders.pkl')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Encode categorical columns matching the notebook
    label_encoders = {}
    cat_cols = ['sex', 'smoker', 'region']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Define feature matrix and target variable in exact column order
    feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    X = df[feature_cols]
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    train_r2 = model_lr.score(X_train, y_train)
    test_r2 = model_lr.score(X_test, y_test)
    print(f"Model trained successfully. Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    joblib.dump(model_lr, model_path)
    joblib.dump(label_encoders, encoders_path)
    print(f"Artifacts saved to:\n  - {model_path}\n  - {encoders_path}")


if __name__ == '__main__':
    train_and_export()

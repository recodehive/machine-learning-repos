import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '50_Startups.csv')
    model_path = os.path.join(base_dir, 'startup.pkl')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load 50 Startups dataset
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode categorical column (State: California -> [0, 0], Florida -> [1, 0], New York -> [0, 1])
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(drop='first'), [3])],
        remainder='passthrough'
    )
    X_transformed = ct.fit_transform(X)

    # Train linear regression model
    regressor = LinearRegression()
    regressor.fit(X_transformed, y)

    # Export trained model
    with open(model_path, 'wb') as f:
        pickle.dump(regressor, f)

    print(f"Model successfully saved to {model_path}")
    print(f"R-squared score: {regressor.score(X_transformed, y):.4f}")


if __name__ == "__main__":
    train_and_export()

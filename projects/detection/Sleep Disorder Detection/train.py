import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import xgboost as xgb


def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'Sleep_health_and_lifestyle_dataset.csv')
    model_path = os.path.join(base_dir, 'model_xgb.pkl')
    encoders_path = os.path.join(base_dir, 'label_encoders.pkl')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Preprocessing matching notebook
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])

    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No Disorder')
    df['Sleep Duration'] = df['Sleep Duration'].round().astype(int)

    # Balance classes via resampling to match RandomOverSampler
    max_size = df['Sleep Disorder'].value_counts().max()
    dfs = []
    for cls in df['Sleep Disorder'].unique():
        cls_df = df[df['Sleep Disorder'] == cls]
        resampled_cls = resample(cls_df, replace=True, n_samples=max_size, random_state=42)
        dfs.append(resampled_cls)
    df = pd.concat(dfs).reset_index(drop=True)

    # Encode categorical columns
    label_encoders = {}
    cat_cols = ['Gender', 'Occupation', 'BMI Category']
    for col in cat_cols:
        le = LabelEncoder()
        if col == 'BMI Category':
            all_vals = list(df[col].unique()) + ['Underweight']
            le.fit(all_vals)
        else:
            le.fit(df[col].unique())
        df[col] = le.transform(df[col])
        label_encoders[col] = le

    # Encode target variable
    target_le = LabelEncoder()
    df['Sleep Disorder'] = target_le.fit_transform(df['Sleep Disorder'])
    label_encoders['Sleep Disorder'] = target_le

    # Split Blood Pressure into Systolic and Diastolic BP
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
    df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
    df = df.drop(columns=['Blood Pressure'])

    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_xgb = xgb.XGBClassifier(random_state=42)
    model_xgb.fit(X_train, y_train)

    train_acc = model_xgb.score(X_train, y_train)
    test_acc = model_xgb.score(X_test, y_test)
    print(f"Model trained successfully. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    joblib.dump(model_xgb, model_path)
    joblib.dump(label_encoders, encoders_path)
    print(f"Artifacts saved to {model_path} and {encoders_path}")


if __name__ == '__main__':
    train_and_export()

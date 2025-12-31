
#? STAGE 2: DATA PREPROCESSING

#* Importing dependencies
import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.components.data_ingestion import datasets


#* Define Preprocessing Function
def preprocess_data(train_df, test_df):
    # Store target variable
    train_target = train_df['smoking']
    test_target = test_df['smoking']
    
    # Remove target from features
    train_features = train_df.drop('smoking', axis=1)
    test_features = test_df.drop('smoking', axis=1)
    
    # Get numeric columns excluding target
    num_cols = train_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle missing values for numeric columns
    imputer = SimpleImputer(strategy='mean')
    train_features[num_cols] = imputer.fit_transform(train_features[num_cols])
    test_features[num_cols] = imputer.transform(test_features[num_cols])

    # Handle categorical values
    cat_cols = train_features.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Encode categorical columns
    if len(cat_cols) > 0:
        train_encoded = pd.DataFrame(
            encoder.fit_transform(train_features[cat_cols]),
            index=train_features.index,
            columns=encoder.get_feature_names_out(cat_cols)
        )
        test_encoded = pd.DataFrame(
            encoder.transform(test_features[cat_cols]),
            index=test_features.index,
            columns=encoder.get_feature_names_out(cat_cols)
        )
        
        # Drop original categorical columns and reset index
        train_features = train_features.drop(cat_cols, axis=1)
        test_features = test_features.drop(cat_cols, axis=1)
        
        # Concatenate encoded features
        train_features = pd.concat([train_features, train_encoded], axis=1)
        test_features = pd.concat([test_features, test_encoded], axis=1)

    # Feature Scaling - only scale numeric columns
    scaler = StandardScaler()
    train_features[num_cols] = scaler.fit_transform(train_features[num_cols])
    test_features[num_cols] = scaler.transform(test_features[num_cols])

    # Split features and target
    X = train_features
    y = train_target

    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )

    # Store selected features
    selected_features = x_train.columns.tolist()

    # Return all 5 expected values
    return x_train, x_val, y_train, y_val, selected_features


def remove_low_variance_features(train_df, test_df, threshold=0.01):
    train_target = train_df['smoking'] if 'smoking' in train_df.columns else None
    train_features = train_df.drop('smoking', axis=1) if 'smoking' in train_df.columns else train_df
    
    test_target = test_df['smoking'] if 'smoking' in test_df.columns else None
    test_features = test_df.drop('smoking', axis=1) if 'smoking' in test_df.columns else test_df
    
    selector = VarianceThreshold(threshold)
    train_features_var = selector.fit_transform(train_features)
    test_features_var = selector.transform(test_features)

    selected_columns = train_features.columns[selector.get_support()]
    
    train_selected = pd.DataFrame(train_features_var, columns=selected_columns, index=train_df.index)
    test_selected = pd.DataFrame(test_features_var, columns=selected_columns, index=test_df.index)
    
    if train_target is not None:
        train_selected['smoking'] = train_target
    if test_target is not None:
        test_selected['smoking'] = test_target
    
    return train_selected, test_selected


def remove_highly_correlated_features(train_df, test_df, threshold=0.9):
    correlation_matrix = train_df.corr()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols)


def select_features_by_mutual_info(train_df, test_df, target_column, num_features=15):
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    mutual_info = mutual_info_classif(X, y, discrete_features='auto')
    feature_scores = pd.Series(mutual_info, index=X.columns)
    selected_features = feature_scores.nlargest(num_features).index.to_list()

    if target_column in test_df.columns:
        return train_df[selected_features + [target_column]], test_df[selected_features + [target_column]]
    else:
        return train_df[selected_features + [target_column]], test_df[selected_features]


def apply_pca(train_df, test_df, n_components=10):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_df)
    test_pca = pca.transform(test_df)
    return pd.DataFrame(train_pca), pd.DataFrame(test_pca)


if __name__ == "__main__":
    #* Load both Train and Test Datasets
    train_ml = pd.DataFrame(datasets["ml-olympiad-smoking"]["train"])
    test_ml = pd.DataFrame(datasets["ml-olympiad-smoking"]["test"])
    train_archive = pd.DataFrame(datasets["archive"]["train"])
    test_archive = pd.DataFrame(datasets["archive"]["test"])

    print("DISPLAY BASIC INFORMATION")
    print("ML Olympiad Train Data Shape:", train_ml.shape)
    print("ML Olympiad Test Data Shape:", test_ml.shape)
    print(train_ml.head())
    print("Archive Train Data Shape:", train_archive.shape)
    print("Archive Test Data Shape:", test_archive.shape)
    print(test_archive.head())

    #* Apply Preprocessing to all datasets
    x_train_ml, x_val_ml, y_train_ml, y_val_ml, selected_features_ml = preprocess_data(train_ml, test_ml)
    x_train_archive, x_val_archive, y_train_archive, y_val_archive, selected_features_archive = preprocess_data(train_archive, test_archive)

    preprocessed_data_paths = {
        "ml-olympiad-smoking": {
            "train": "Y:/SmokingML V2/data/processed/ml_olympiad_train.csv",
            "test": "Y:/SmokingML V2/data/processed/ml_olympiad_test.csv"
        },
        "archive": {
            "train": "Y:/SmokingML V2/data/processed/archive_train.csv",
            "test": "Y:/SmokingML V2/data/processed/archive_test.csv"
        }
    }

    for dataset_name, paths in preprocessed_data_paths.items():
        for key, path in paths.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)

    pd.concat([x_train_ml, y_train_ml], axis=1).to_csv(preprocessed_data_paths["ml-olympiad-smoking"]["train"], index=False)
    pd.concat([x_val_ml, y_val_ml], axis=1).to_csv(preprocessed_data_paths["ml-olympiad-smoking"]["test"], index=False)
    pd.concat([x_train_archive, y_train_archive], axis=1).to_csv(preprocessed_data_paths["archive"]["train"], index=False)
    pd.concat([x_val_archive, y_val_archive], axis=1).to_csv(preprocessed_data_paths["archive"]["test"], index=False)

    print("Preprocessed data has been saved successfully!")

    #* Variance Thresholding
    preprocessed_train_ml, preprocessed_test_ml = remove_low_variance_features(pd.concat([x_train_ml, y_train_ml], axis=1), pd.concat([x_val_ml, y_val_ml], axis=1))
    preprocessed_train_archive, preprocessed_test_archive = remove_low_variance_features(pd.concat([x_train_archive, y_train_archive], axis=1), pd.concat([x_val_archive, y_val_archive], axis=1))

    #* Feature Selection
    preprocessed_train_ml, preprocessed_test_ml = select_features_by_mutual_info(preprocessed_train_ml, preprocessed_test_ml, target_column='smoking')
    preprocessed_train_archive, preprocessed_test_archive = select_features_by_mutual_info(preprocessed_train_archive, preprocessed_test_archive, target_column='smoking')

    #* ✅ Optional assertion checks
    assert 'smoking' in preprocessed_train_ml.columns, "Target column 'smoking' missing in training set!"
    assert 'smoking' in preprocessed_test_ml.columns, "Target column 'smoking' missing in test set!"
    assert 'smoking' in preprocessed_train_archive.columns, "Target column 'smoking' missing in archive training set!"
    assert 'smoking' in preprocessed_test_archive.columns, "Target column 'smoking' missing in archive test set!"

    #* ✅ Debug: Show absolute save paths
    print("\n✅ Saving preprocessed files to:")
    print("ML Train Path      :", os.path.abspath("Y:/SmokingML V2/data/processed/train_ml.csv"))
    print("ML Test Path       :", os.path.abspath("Y:/SmokingML V2/data/processed/test_ml.csv"))
    print("Archive Train Path :", os.path.abspath("Y:/SmokingML V2/data/processed/train_archive.csv"))
    print("Archive Test Path  :", os.path.abspath("Y:/SmokingML V2/data/processed/test_archive.csv"))

    #* Save final preprocessed files
    preprocessed_train_ml.to_csv("Y:/SmokingML V2/data/processed/train_ml.csv", index=False)
    preprocessed_test_ml.to_csv("Y:/SmokingML V2/data/processed/test_ml.csv", index=False)
    preprocessed_train_archive.to_csv("Y:/SmokingML V2/data/processed/train_archive.csv", index=False)
    preprocessed_test_archive.to_csv("Y:/SmokingML V2/data/processed/test_archive.csv", index=False)

    print("Feature Engineering and Selection completed Successfully!")


    import json

    #* Save selected features to JSON for both datasets
    selected_features_dir = "Y:/SmokingML V2/artifacts/models"
    os.makedirs(selected_features_dir, exist_ok=True)

    # Remove 'smoking' from selected columns before saving (optional based on use-case)
    selected_columns_olympiad = [col for col in preprocessed_train_ml.columns if col != 'smoking']
    selected_columns_archive = [col for col in preprocessed_train_archive.columns if col != 'smoking']

    # Save to JSON
    with open(os.path.join(selected_features_dir, "feature_columns_olympiad.json"), "w") as f:
        json.dump(selected_columns_olympiad, f, indent=4)

    with open(os.path.join(selected_features_dir, "feature_columns_archive.json"), "w") as f:
        json.dump(selected_columns_archive, f, indent=4)

    print("✅ Feature columns JSON files saved successfully!")

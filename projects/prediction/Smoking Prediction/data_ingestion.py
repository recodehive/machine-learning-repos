
#? STAGE 1: DATA INGESTION

import os
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

class DataIngestion:
    def __init__(self,dataset_paths):
        """
        dataset_paths: dictionary containing dataset paths as keys and their paths as values
        Example: 
        {
            "dataset1": {"train": "path/to/dataset1_train.csv", "test": "path/to/dataset1_test.csv"}
            "dataset2": {"train": "path/to/dataset2_train.csv", "test": "path/to/dataset2_test.csv"}
        }
        """
        self.dataset_paths = dataset_paths

    def load_data(self):
        datasets = {}
        for dataset_name, paths in self.dataset_paths.items():
            # Load training data
            train_df = pd.read_csv(paths["train"])
            
            # Split into train and test
            train_data, test_data = train_test_split(
                train_df, test_size=0.2, random_state=42
            )
            
            # Store in nested structure
            datasets[dataset_name] = {
                "train": train_data,
                "test": test_data
            }
        
        return datasets

dataset_paths = {
    "ml-olympiad-smoking": {
        "train": "Y:/SmokingML V2/data/raw/ml-olympiad-smoking/train.csv"
    },
    "archive": {
        "train": "Y:/SmokingML V2/data/raw/archive/train_dataset.csv"
    }
}

# Create data ingestion object and load data
data_ingestion = DataIngestion(dataset_paths)
datasets = data_ingestion.load_data()

# Now we can safely access the train/test splits
print("ML Olympiad Training Data Type:", type(datasets["ml-olympiad-smoking"]["train"]))
print("ML Olympiad Training Data Shape:", datasets["ml-olympiad-smoking"]["train"].shape)
print("Archive Training Data Type:", type(datasets["archive"]["train"]))
print("Archive Training Data Shape:", datasets["archive"]["train"].shape)

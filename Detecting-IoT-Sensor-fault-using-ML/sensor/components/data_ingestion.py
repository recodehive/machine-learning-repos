import os
import sys 
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sensor.constant.training_pipeline import SCHEMA_DROP_COLS, SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.config_entity import DataIngestionConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.data_access.sensor_data import SensorData
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
class DataIngestion: 
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._scgema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e,sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """Export MongoDB collection record as DataFrame into feature

            Returns:
                DataFrame: dataset for the project
        """    

        logging.info(f"Exporting data from MongoDB to feature store")

        sensor_data = SensorData()
        print(self.data_ingestion_config.collection_name)
        dataframe = sensor_data.export_collection_as_dataframe(
            collection_name=self.data_ingestion_config.collection_name
        )

        logging.info(f"Shape of dataframe: {dataframe.shape}")

        # Creating Folder 
        feature_store_file_path = self.data_ingestion_config.feature_store_file_path

        dir_path = os.path.dirname(feature_store_file_path)

        os.makedirs(dir_path, exist_ok=True) # if folder not available create it else don't

        logging.info(
            f"Saving exported data into feature store file path: {feature_store_file_path}"
        )

        dataframe.to_csv(feature_store_file_path, index=False, header=True)

        return dataframe
                
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:

        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set,test_set = train_test_split(
                dataframe,test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
        
        except Exception as e:
            raise SensorException(e,sys)

        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            dataframe = dataframe.drop(self._scgema_config['drop_columns'],axis=1)

            logging.info(f"Shape of dataframe after dropping rows: {dataframe.shape}")

            logging.info("Got the data from MongoDB")
        
            self.split_data_as_train_test(dataframe=dataframe)

            logging.info("Performed Train Test Split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e,sys)
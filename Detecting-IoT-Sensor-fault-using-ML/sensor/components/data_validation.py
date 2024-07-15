import os,sys
from scipy.stats import ks_2samp
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils.main_utils import read_yaml_file, write_yaml_file 
import pandas as pd

class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e,sys)

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])

            if(len(dataframe.columns) == number_of_columns):
                return True

            return False
            pass
        except Exception as e:
            raise SensorException(e,sys)

    def drop_zero_std_columns(self,dataframe:pd.DataFrame):
        pass

    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns
            numerical_column_present = True
            misssing_numerical_column = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    misssing_numerical_column.append(num_column)
            logging.info(f"Missing muerical columns: [{misssing_numerical_column}]")
            return numerical_column_present
        except Exception as e:
            raise SensorException(e,sys)


    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e,sys)


    def detect_dataset_drift(self,base_df: pd.DataFrame,current_df: pd.DataFrame
                                ,threshold=0.5)->bool:
        try:
            status = True
            report={}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    status = False
                    is_found=True
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})

            drift_report_file_path=self.data_validation_config.drift_report_file_path

            # Create Directory
            dir_path=os.path.dirname(drift_report_file_path)

            os.makedirs(dir_path,exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path,content=report)
            return status



        except Exception as e:
            raise SensorException(e,sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info("Starting Data Validation")

            error_message = ""
            
            train_file_path = self.data_ingestion_artifact.trained_file_path
            
            test_file_path = self.data_ingestion_artifact.test_file_path

            # reading data from train and test file location


            #print(train_file_path)
            


            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)


            #print(type(train_dataframe))


            # Validate number of columns in training dataframe
            status = self.validate_number_of_columns(dataframe=train_dataframe)

            if not status:
                error_message = f"{error_message}Train dataframe does not contain all columns.\n"
            else:
                logging.info("Train dataframe contain all columns.")


            # Validate number of columns in testing dataframe
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain all columns.\n"
            else:
                logging.info("Train dataframe contain all columns.")

            

            # Validate numerical columns in training dataframe
            status = self.is_numerical_column_exist(dataframe = train_dataframe)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain all numerical columns.\n"
            else:
                logging.info("Train dataframe contain all numerical columns")


            # Validate numerical columns in testing dataframe
            status = self.is_numerical_column_exist(dataframe = test_dataframe)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain all numerical columns.\n"
            else:
                logging.info("Test dataframe contain all numerical columns")


            # If there is a huge error like one or more column/columns is missing then, give exception
            if(len(error_message)>0):
                raise Exception(error_message)
        
        
            # Checking data draft
            
            # We will be running the training pipeline even if we detect data drift
            status = self.detect_dataset_drift(base_df=train_dataframe,current_df=train_dataframe)


            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            return data_validation_artifact

        except Exception as e:
            raise SensorException(e,sys)


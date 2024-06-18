import sys,os
from os.path import dirname, join, abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))
from logger import logging
from exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.intiate_data_ingestion()
    print(train_data_path,test_data_path)
    data_transformation =  DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
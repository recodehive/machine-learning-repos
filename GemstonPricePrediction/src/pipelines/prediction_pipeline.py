import sys,os
from os.path import dirname, join, abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))

from exception import CustomException
from logger import logging
from utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path= os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 carat:float,
                 depth: float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity


    def get_data_as_datarframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
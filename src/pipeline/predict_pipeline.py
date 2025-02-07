import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer           

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Ensure features is a DataFrame
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features, columns=["odor", "gill_color", "spore_print_color", "gill_size", "bruises"])  # Adjust column names as needed

            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform features using the preprocessor
            data_transformed = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_transformed)
            
            return preds[0]
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 odor,
                 gill_color, 
                 spore_print_color, 
                 gill_size,
                 bruises
    ):
        self.odor = odor
        self.gill_color = gill_color
        self.spore_print_color = spore_print_color
        self.gill_size = gill_size
        self.bruises = bruises    
    
    def get_data_as_data_frame(self):
        try:
            custome_data_input_dict = {
                'odor': [self.odor],
                'gill_color': [self.gill_color],
                'spore_print_color': [self.spore_print_color],
                'gill_size': [self.gill_size],
                'bruises': [self.bruises]
            }
            return pd.DataFrame(custome_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
 
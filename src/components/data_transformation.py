import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import joblib

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTranformation:
    '''
    This function is responsible for data transformation.
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = ["odor", "gill_color", "spore_print_color", "gill_size", "bruises"]
            
            # numerical_columns = []

            # num_pipeline = Pipeline(
            #     steps=[
            #         ("Imputer", SimpleImputer(strategy="median")),
            #         ("Scaler", StandardScaler())]

            # )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("OrdinalEncoder", OrdinalEncoder())
                ]
            )
            
            logging.info(f"Categorical Columns {categorical_columns} encoding completed.")

            preprocessor = ColumnTransformer([
                # ('num_pipeline', num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'class'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            target_encoder = LabelEncoder()
            target_feature_train_arr = target_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = target_encoder.transform(target_feature_test_df)

            train_input_df = pd.DataFrame(data=input_feature_train_arr, columns=input_feature_train_df.columns)
            test_input_df = pd.DataFrame(data=input_feature_test_arr, columns=input_feature_test_df.columns)
            train_target_df = pd.DataFrame(data=target_feature_train_arr, columns=[target_column_name])
            test_target_df = pd.DataFrame(data=target_feature_test_arr, columns=[target_column_name])

            train_df = pd.concat([train_input_df, train_target_df], axis=1)
            test_df = pd.concat([test_input_df, test_target_df], axis=1)
            
            # logging.info(f"Saved preprocessing object\n{train_df}\n{test_df}")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_df,
                test_df,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
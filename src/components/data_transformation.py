import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler

# class CustomException(Exception):
#     def __init__(self, message, original_exception):
#         super().__init__(f"{message}: {original_exception}")
#         self.original_exception = original_exception

# try:
#     # Assuming 'dataframe' is already defined and loaded with data
#     # Example columns to be scaled
#     columns_to_scale = ['feature1', 'feature2']
    
#     # Initialize the scaler
#     scaler = StandardScaler()
    
#     # Correct usage of fit_transform
#     scaled_data = scaler.fit_transform(dataframe[columns_to_scale])
    
# except TypeError as e:
#     raise CustomException("fit_transform() takes 2 positional arguments but 3 were given", e)
# except Exception as e:
#     raise CustomException("An error occurred during data transformation", e)


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
            categorical_columns = [
                                   'cap-shape',
                                   'cap-surface',
                                   'cap-color',
                                   'bruises',
                                   'odor',
                                   'gill-attachment',
                                   'gill-spacing',
                                   'gill-size',
                                   'gill-color',
                                   'stalk-shape',
                                   'stalk-root',
                                   'stalk-surface-above-ring',
                                   'stalk-surface-below-ring',
                                   'stalk-color-above-ring',
                                   'stalk-color-below-ring',
                                   'veil-type',
                                   'veil-color',
                                   'ring-number',
                                   'ring-type',
                                   'spore-print-color',
                                   'population',
                                   'habitat']
            
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

            # train_arr = np.c_[
            #     input_feature_train_arr,
            #     target_feature_train_arr
            # ]

            # test_arr = np.c_[ 
            #     input_feature_test_arr,
            #     target_feature_test_arr
            # ]

            train_input_df = pd.DataFrame(data=input_feature_train_arr, columns=input_feature_train_df.columns)
            test_input_df = pd.DataFrame(data=input_feature_test_arr, columns=input_feature_test_df.columns)
            train_target_df = pd.DataFrame(data=target_feature_train_arr, columns=[target_column_name])
            test_target_df = pd.DataFrame(data=target_feature_test_arr, columns=[target_column_name])

            train_df = pd.concat([train_input_df, train_target_df], axis=1)
            test_df = pd.concat([test_input_df, test_target_df], axis=1)
            
            logging.info(f"Saved preprocessing object\n{train_df}\n{test_df}")

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
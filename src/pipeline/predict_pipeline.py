import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, cap_shape, 
                 cap_surface, 
                 cap_color, 
                 bruises,
                 odor, 
                 gill_attachment, 
                 gill_spacing, 
                 gill_size, 
                 gill_color, 
                 stalk_shape, 
                 stalk_root, 
                 stalk_surface_above_ring, 
                 stalk_surface_below_ring, 
                 stalk_color_above_ring, 
                 stalk_color_below_ring, 
                 veil_type, 
                 veil_color, 
                 ring_number, 
                 ring_type, 
                 spore_print_color, 
                 population, 
                 habitat):
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size= gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat
    
    def get_data_as_data_frame(self):
        try:
            custome_data_input_dict = {
                'cap-shape': [self.cap_shape],
                'cap-surface': [self.cap_surface],
                'cap-color': [self.cap_color],
                'bruises': [self.bruises],
                'odor': [self.odor],
                'gill-attachment': [self.gill_attachment],
                'gill-spacing': [self.gill_spacing],
                'gill-size': [self.gill_size],
                'gill-color': [self.gill_color],
                'stalk-shape':[ self.stalk_shape],
                'stalk-root': [self.stalk_root],
                'stalk-surface-above-ring': [self.stalk_surface_above_ring],
                'stalk-surface-below-ring': [self.stalk_surface_below_ring],
                'stalk-color-above-ring': [self.stalk_color_above_ring],
                'stalk-color-below-ring': [self.stalk_color_below_ring],
                'veil-type': [self.veil_type],
                'veil-color': [self.veil_color],
                'ring-number': [self.ring_number],
                'ring-type': [self.ring_type],
                'spore-print-color': [self.spore_print_color],
                'population': [self.population],
                'habitat': [self.habitat]
            }
            return pd.DataFrame(custome_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

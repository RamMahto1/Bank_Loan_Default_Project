from src.utils import load_obj
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys

class PredictPipeline:
    def __init__(self, preprocessor_path, model_path):
        self.preprocessor = load_obj(preprocessor_path)    
        self.model = load_obj(model_path)    

    def predict(self, input_data):
        try:
            # Step 1: Convert input to DataFrame
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data

            logging.info("Applying preprocessor on input data")
            processed_data = self.preprocessor.transform(input_df)  # fixed transform()

            logging.info("Predicting with loaded model")
            prediction = self.model.predict(processed_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)

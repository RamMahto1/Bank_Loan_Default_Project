from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys

@dataclass

class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")
    
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
        
    def initialize_data_ingestion(self):
        try:
            data=os.path.join("notebook","combined_cleaned_data.csv")
            df = pd.read_csv(data)
            logging.info("read the data as data frame")
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("spliting the data into train test split")
            train_set,test_set = train_test_split(df,test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("data ingestion complete")
            
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
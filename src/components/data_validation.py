from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass
import pandas as pd

class DataValidation:
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
        
    def initialize_data_validation(self):
        try:
            train_df=pd.read_csv(self.train_data)
            test_df = pd.read_csv(self.test_data)
            
            # check the shape of data
            logging.info(f"train data shape: \n{train_df.shape}")
            logging.info(f"test data shape:\n{test_df.shape}")
            
            
            ## checking null value
            logging.info(f"train data null value: \n{train_df.isnull().sum()}")
            logging.info(f"test data null value:\n{test_df.isnull().sum()}")
            
            ## data information
            logging.info("train data info")
            train_df.info()
            
            logging.info("test data info")
            test_df.info()
            
            ## checking duplicate 
            logging.info(f"train data duplicate: \n{train_df.duplicated().sum()}")
            logging.info(f"test data dupliated:\n{test_df.duplicated().sum()} ")
            
            ## statatics test 
            logging.info(f"train data stastatics info:{train_df.describe()}")
            logging.info(f"test data info:{test_df.describe()}")
            
            expected_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','AMT_INCOME_TOTAL',
                                'AMT_CREDIT','DAYS_BIRTH','PREV_AMT_APPLICATION_MEAN','PREV_AMT_CREDIT_MEAN',
                                'NUM_PREV_LOANS','HAS_PREV_LOANS']
            
            missing_columns = [col for col in expected_columns if col not in train_df.columns]
            
            if missing_columns:
                raise CustomException(f"Missing columns in train data:{missing_columns}",sys)
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    
    
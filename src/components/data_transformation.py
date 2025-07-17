from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import saved_obj
from imblearn.over_sampling import SMOTE


@dataclass
class DataTransformationConfig:
    preprocessor_file_path_obj:str = os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_data_transformer_obj(self):
            try:
                numerical_features = ['AMT_INCOME_TOTAL','AMT_CREDIT','DAYS_BIRTH','PREV_AMT_APPLICATION_MEAN',
                                      'PREV_AMT_CREDIT_MEAN','NUM_PREV_LOANS','HAS_PREV_LOANS']
                categorical_features = ['NAME_CONTRACT_TYPE','CODE_GENDER']
                
                
                num_pipeline = Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler())
                    ]
                )
                cat_pipeline = Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))
                    ]
                )
                
                preprocessor = ColumnTransformer(
                    [
                        ("num",num_pipeline,numerical_features),
                        ("cat",cat_pipeline,categorical_features)
                    ]
                )
                
                logging.info(f"Numerical feature: {numerical_features}")
                logging.info(f"Categorical feature: {categorical_features}")
                return preprocessor
            except Exception as e:
                raise CustomException(e,sys)
            
            
    def initialize_data_transformation(self,train_arr,test_arr):
        
        try:
            train_df = pd.read_csv(train_arr)
            test_df = pd.read_csv(test_arr)
            logging.info("read data as data frame")
            logging.info("obtaining preprocessor file path")
            
            target_columns = ['TARGET']
            
            preprocessor_obj = self.get_data_transformer_obj()
            
            input_feature_train_df = train_df.drop(columns=target_columns)
            target_feature_train_df = train_df[target_columns]
            
            
            input_feature_test_df = test_df.drop(columns=target_columns)
            target_feature_test_df = test_df[target_columns]
            
            logging.info(f"Applying preprocessor object on training and test dataset")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
            input_feature_train_arr, target_feature_train_df.values.ravel()
        )
            logging.info(f"Before SMOTE: {np.bincount(target_feature_train_df.values.ravel())}")
            logging.info(f"After SMOTE: {np.bincount(y_train_resampled)}")

            
            train_arr = np.c_[X_train_resampled,y_train_resampled]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df.values]
            
            saved_obj(
                file_path = self.data_transformation_config.preprocessor_file_path_obj,
                obj=preprocessor_obj
                
            )
            
            return(
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_file_path_obj
            )
            
        except Exception as e:
            raise CustomException(e,sys)
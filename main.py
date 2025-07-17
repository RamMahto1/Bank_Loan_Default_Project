from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import saved_obj
from src.components.data_validation import DataValidation
#logging.info("logger setup complete")


# try:
#     10/0
# except Exception as e:
#     raise CustomException(e,sys)
if __name__=="__main__":
    try:
        # step 1 : Data Ingestion
        Data_ingestion=DataIngestion()
        train_data,test_data = Data_ingestion.initialize_data_ingestion()
        print(f"train data:{train_data}")
        print(f"test data:{test_data}")
        
        
        ## Step 2: Data Transformation
        data_transformer=DataTransformation()
        train_arr,test_arr,_=data_transformer.initialize_data_transformation(train_arr=train_data,test_arr=test_data)
        logging.info(f"Data transformation train: {train_arr}")
        logging.info(f"Data transfomation test:{test_arr}")
        
        # Step 3: Data validation
        data_validation=DataValidation(train_data,test_data)
        data_validation.initialize_data_validation()
        logging.info("Data validation completed")
        
        
        
    except Exception as e:
        raise CustomException(e,sys)

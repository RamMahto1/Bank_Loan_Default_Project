from src.logger import logging
from src.exception import CustomException
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initialize_data_ingestion()
        print(f"Train file path: {train_path}")
        print(f"Test file path: {test_path}")
        logging.info("Data ingestion complete")

        # Step 2: Data Transformation
        data_transformer = DataTransformation()
        X_train_resampled, y_train_resampled, X_test, y_test = data_transformer.initialize_data_transformation(
            train_file_path=train_path,
            test_file_path=test_path
        )
        logging.info(f"Transformed X_train shape: {X_train_resampled.shape}")
        logging.info(f"Transformed y_train shape: {y_train_resampled.shape}")
        logging.info(f"Transformed X_test shape: {X_test.shape}")
        logging.info(f"Transformed y_test shape: {y_test.shape}")
        logging.info("Data transformation complete")

        # Step 3: Data Validation
        data_validation = DataValidation(train_data=train_path, test_data=test_path)
        data_validation.initialize_data_validation()
        logging.info("Data validation complete")

        # Step 4: Model Training
        model_trainer = ModelTrainer()
        report, best_model_name, best_model = model_trainer.initialize_model_trainer(X_train_resampled, y_train_resampled, X_test, y_test)
        import pprint
        print("Executing model training step...")




        logging.info(f"Model training complete. Best model: {best_model_name}")
        #print("Model Training Report:", report)
        pprint.pprint(report)

    except Exception as e:
        raise CustomException(e, sys)

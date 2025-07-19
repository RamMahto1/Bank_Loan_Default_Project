from src.logger import logging
from src.exception import CustomException
import sys
import yaml
import pandas as pd
import numpy as np
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from mlflow_experiment.mlflow_config import MLflowLogger
from src.utils import save_model
import mlflow.sklearn  # for logging sklearn model
from mlflow.models.signature import infer_signature

if __name__ == "__main__":
    try:
        # Load mlflow config
        with open("mlflow_experiment/mlflow_logger.yaml", "r") as f:
            config = yaml.safe_load(f)

        mlflow_logger = MLflowLogger(
            experiment_name=config["experiment_name"],
            tracking_uri=config["tracking_uri"]
        )

        # Start MLflow run
        with mlflow_logger.start_run(run_name="Loan Default Training"):

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
            report, best_model_name, best_model = model_trainer.initialize_model_trainer(
                X_train_resampled, y_train_resampled, X_test, y_test
            )
            import pprint
            print("Executing model training step...")

            logging.info(f"Model training complete. Best model: {best_model_name}")
            pprint.pprint(report)

            # Save best model locally
            save_model(best_model, "artifacts/model/model.pkl")

            # Log metrics to MLflow
            mlflow_logger.log_metrics({
                "accuracy": report.get("accuracy", 0),
                "precision": report.get("precision", 0),
                "recall": report.get("recall", 0),
                "f1_score": report.get("f1_score", 0)
            })

            # Convert X_test to DataFrame for MLflow logging
            X_test_df = pd.DataFrame(X_test)

            # Log model itself to MLflow with signature and input example
            signature = infer_signature(X_test_df, best_model.predict(X_test_df))

            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                input_example=X_test_df.iloc[:5]
            )

    except Exception as e:
        raise CustomException(e, sys)

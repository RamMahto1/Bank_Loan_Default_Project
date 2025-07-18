import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.utils import saved_obj, model_evaluate


@dataclass
class ModelTrainerConfig:
    model_obj_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initialize_model_trainer(self, X_train_resampled, y_train_resampled, X_test, y_test):
        try:
            
            
            models = {
                "LogisticRegression": LogisticRegression(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "SupportVectorClassifier": SVC(),
                "KNeighborsClassifier": KNeighborsClassifier()
            }

            params = {
                "LogisticRegression": {
                    'penalty': ['l2'],
                    'C': [1.0],
                    'random_state': [42]
                },
                "DecisionTreeClassifier": {
                    'criterion': ['gini'],
                    'max_depth': [3, 5, None],
                    'min_samples_split': [2, 3, 5],
                    'min_samples_leaf': [1],
                    'random_state': [42]
                },
                "RandomForestClassifier": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 6, 8],
                    'max_leaf_nodes': [6],
                    'max_features': ['sqrt']
                },
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.01, 0.02],
                    'random_state': [42]
                },
                "GradientBoostingClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.01, 0.02],
                    'max_depth': [3],
                    'random_state': [42]
                },
                "SupportVectorClassifier": {
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [5],
                    'weights': ['uniform'],
                    'algorithm': ['auto'],
                    'leaf_size': [30],
                    'p': [2]
                }
            }

            # Evaluate models using the resampled data
            report, best_model_name, best_model = model_evaluate(
                X_train_resampled,
                y_train_resampled,
                X_test,
                y_test,
                models,
                params
            )

            logging.info(f"Best model found: {best_model_name}")

            # Save best model
            saved_obj(
                file_path=self.model_trainer_config.model_obj_file_path,
                obj=best_model
            )

            return report, best_model_name, best_model

        except Exception as e:
            raise CustomException(e, sys)

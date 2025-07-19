import pickle
import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from src.exception import CustomException
from src.logger import logging
import joblib


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved at: {file_path}")


def saved_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def model_evaluate(X_train_resampled, y_train_resampled, X_test, y_test, models, params):
    try:
        report = {}
        best_score = float('-inf')
        best_model_name = None
        best_model = None

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            param = params.get(model_name, {})
            gs = GridSearchCV(model, param_grid=param, cv=5, scoring='f1_weighted')
            gs.fit(X_train_resampled, y_train_resampled)
            best_model_candidate = gs.best_estimator_
            y_pred = best_model_candidate.predict(X_test)

            f1score = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            acc = accuracy_score(y_test, y_pred)

            report[model_name] = {
                'f1_score': f1score,
                'precision_score': precision,
                'recall_score': recall,
                'accuracy': acc
            }

            if f1score > best_score:
                best_score = f1score
                best_model_name = model_name
                best_model = best_model_candidate

        return report, best_model_name, best_model

    except Exception as e:
        raise CustomException(e, sys)


def load_obj(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

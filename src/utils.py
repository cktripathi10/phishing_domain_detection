import os
import sys

import numpy as np 
import pandas as pd
import dill  # Using dill for potentially more complex objects
from sklearn.metrics import accuracy_score  # Assuming classification task
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException  # Custom exception handling

def save_object(obj, file_path):
    """ Saves an object to a specified path using dill. """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Ensure directory exists

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Use dill to dump the object

    except Exception as e:
        raise CustomException(str(e), sys.exc_info())  # Enhanced exception information

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """ Evaluates models using GridSearchCV and returns a report with model performance metrics. """
    try:
        report = {}

        for model_name, model in models.items():
            param = params[model_name]
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            # Best model retraining
            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Model performance
            train_model_score = accuracy_score(y_train, y_train_pred)  # Assuming accuracy metric
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'best_params': gs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(str(e), sys.exc_info())

def load_object(file_path):
    """ Loads an object from a specified path using dill. """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)  # Use dill to load the object

    except Exception as e:
        raise CustomException(str(e), sys.exc_info())


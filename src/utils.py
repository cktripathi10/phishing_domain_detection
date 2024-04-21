import os
import sys
import dill
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(obj, file_path):
    """Saves an object to a specified path using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(str(e), sys.exc_info())

def evaluate_models(X_train, y_train, X_test, y_test, models, params, scorers):
    """Evaluates models using GridSearchCV with multiple scoring functions and returns a report with model performance metrics."""
    try:
        report = {}
        if not isinstance(scorers, dict):
            raise ValueError("Scorers must be a dictionary of scoring functions.")

        for model_name, model in models.items():
            param = params[model_name]
            if not param:
                raise ValueError(f"No parameters defined for {model_name}")

            gs = GridSearchCV(model, param, scoring=scorers, refit=list(scorers.keys())[0], cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            # Use GridSearchCV's scoring results instead of manually scoring
            report[model_name] = {
                'best_score': gs.best_score_,
                'test_score': gs.score(X_test, y_test),
                'best_params': gs.best_params_,
                'scorer_names': list(scorers.keys())
            }

        return report

    except Exception as e:
        raise CustomException(str(e), sys.exc_info())

def load_object(file_path):
    """Loads an object from a specified path using dill."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(str(e), sys.exc_info())

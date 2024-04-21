import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                #"Gradient Boosting": GradientBoostingClassifier(),
                #"XGB Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                #"CatBoost Classifier": CatBoostClassifier(verbose=False),
                #"AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME')
            }
            params = {
                "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [10, 15, 20, None]},
                "Random Forest": {'n_estimators': [50, 100, 200], 'max_features': ['sqrt'], 'criterion': ['gini', 'entropy']},
                #"Gradient Boosting": {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300]},
                #"XGB Classifier": {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]},
                #"CatBoost Classifier": {'depth': [4, 6, 10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [50, 100, 200]},
                #"AdaBoost Classifier": {'learning_rate': [0.01, 0.1, 0.5], 'n_estimators': [50, 100, 200]}
            }
            scorers = {
                'AUC': make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovo'),
                'F1': make_scorer(f1_score, average='weighted')
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                           models=models, params=params, scorers=scorers)

            best_model_score = max(model_report.values(), key=lambda x: x['test_score'])
            best_model_name = next(name for name, report in model_report.items() if report['test_score'] == best_model_score['test_score'])
            best_model = models[best_model_name]

            logging.info(f"Training {best_model_name} with best parameters.")
            best_model.fit(X_train, y_train)  # Ensure model is retrained on the entire training set

            if best_model_score['test_score'] < 0.85:  # Adjust the threshold as needed
                raise CustomException("No best model found with sufficient score.")
            logging.info(f"Best model found: {best_model_name} with score {best_model_score['test_score']}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = f1_score(y_test, predicted, average='weighted')
            roc_auc = roc_auc_score(y_test, predicted, average='macro', multi_class='ovo')
            return accuracy, roc_auc

        except Exception as e:
            logging.error("Error during model training or prediction: {}".format(e), exc_info=True)
            raise CustomException(e, sys)


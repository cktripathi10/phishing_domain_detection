import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self, train_data_path):
        """
        Create a data transformation pipeline dynamically based on the dataset structure.
        """
        try:
            df = pd.read_csv(train_data_path, nrows=1)
            
            # Determine columns based on data type
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Removing the target column if included accidentally in features
            if 'phishing' in numerical_columns:
                numerical_columns.remove('phishing')
            if 'phishing' in categorical_columns:
                categorical_columns.remove('phishing')

            # Setup pipelines for numerical and categorical data
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combining pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info(f"Configured preprocessing with numerical columns: {numerical_columns} and categorical columns: {categorical_columns}")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Starting data transformation process.")

            preprocessing_obj = self.get_data_transformer_object(train_path)

            # Transforming data
            input_feature_train_arr = preprocessing_obj.fit_transform(train_df.drop(columns=['phishing']))
            input_feature_test_arr = preprocessing_obj.transform(test_df.drop(columns=['phishing']))

            # Combining features with target for training set
            train_arr = np.c_[input_feature_train_arr, train_df['phishing']]
            test_arr = np.c_[input_feature_test_arr, test_df['phishing']]

            logging.info("Data transformation completed and saved.")

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)

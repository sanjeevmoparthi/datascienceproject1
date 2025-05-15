# from src.datascienceproject1.logger import logging
# from src.datascienceproject1.exception import CustomException
# from src.datascienceproject1.components.data_ingestion import DataIngestion
# from src.datascienceproject1.components.data_ingestion import DataIngestionConfig
# from src.datascienceproject1.components.data_transformation import DataTransformationConfig,DataTransformation
# from src.datascienceproject1.components.model_tranier import ModelTrainer,ModelTrainerConfig
# import sys


# if __name__=="__main__":
#     logging.info("The execution has started")

#     try:
#         #data ingestion
#         data_ingestion=DataIngestion()
#         train_data_path,test_data_path =  data_ingestion.initiate_data_ingestion()

#         # data transformation
#         data_transformation_config = DataTransformationConfig()
#         data_transformation = DataTransformation()
#         train_arr,test_arr,_= data_transformation.initiate_data_transormation(train_data_path,test_data_path)


#         ### MOdel training:
#         model_trainer = ModelTrainer()
#         print( model_trainer.initiate_model_trainer(train_arr,test_arr))

#     except Exception as e:
#         logging.info("Custom Exception")
#         raise CustomException(e,sys)
from dataclasses import dataclass
import os
import sys
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.datascienceproject1.exception import CustomException
from src.datascienceproject1.logger import logging
from src.datascienceproject1.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = mean_squared_error(actual, pred, squared=False)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with r2_score >= 0.6")

            logging.info(f"Best model: {best_model_name}")

            best_params = params.get(best_model_name, {})

            mlflow.set_registry_uri("https://dagshub.com/sanjeevmoparthi/datascienceproject1.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            final_r2_score = r2_score(y_test, best_model.predict(X_test))
            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)


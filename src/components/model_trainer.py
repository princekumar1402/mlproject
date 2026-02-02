import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),

                # ===== ONLY CATBOOST PART CHANGED =====
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42,
                    thread_count=-1,
                    loss_function='RMSE',
                    eval_metric='R2',
                    early_stopping_rounds=50
                ),
                # ======================================

                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },

                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Linear Regression":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                # ===== ONLY CATBOOST PARAMS CHANGED =====
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.03, 0.05],
                    'iterations': [300, 700, 1200],
                    'l2_leaf_reg': [1, 3, 5]
                },
                # =========================================

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score greater than 0.6", sys)

            logging.info("Saving the best model")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,        
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted) 
            return r2_square
        
        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)

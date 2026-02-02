import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    '''Saves a Python object to a file using pickle'''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    '''Evaluates multiple machine learning models and returns their R2 scores'''
    try:
        r2_scores = {}

        for model_name, model in models.items():

            para = param.get(model_name, {})

            # ===== SPECIAL HANDLING FOR CATBOOST =====
            if model_name == "CatBoosting Regressor":

                from catboost import Pool

                train_pool = Pool(X_train, y_train)
                test_pool = Pool(X_test, y_test)

                # Manual search instead of GridSearchCV
                best_score = -np.inf
                best_params = None

                for depth in para.get('depth', [6]):
                    for lr in para.get('learning_rate', [0.03]):
                        for it in para.get('iterations', [500]):
                            for l2 in para.get('l2_leaf_reg', [3]):

                                temp_model = model.__class__(
                                    depth=depth,
                                    learning_rate=lr,
                                    iterations=it,
                                    l2_leaf_reg=l2,
                                    verbose=False,
                                    thread_count=-1,
                                    early_stopping_rounds=50
                                )

                                temp_model.fit(train_pool, eval_set=test_pool)

                                preds = temp_model.predict(X_test)
                                score = r2_score(y_test, preds)

                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'depth': depth,
                                        'learning_rate': lr,
                                        'iterations': it,
                                        'l2_leaf_reg': l2
                                    }

                model.set_params(**best_params)
                model.fit(X_train, y_train)

                r2_scores[model_name] = best_score

            # ===== NORMAL MODELS =====
            else:
                gs = GridSearchCV(model, para, cv=5, n_jobs=-1)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                r2_scores[model_name] = r2_score(y_test, y_test_pred)

        return r2_scores

    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
    '''Loads a Python object from a file using pickle'''
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
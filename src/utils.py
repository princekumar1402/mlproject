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
    
def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    '''Evaluates multiple machine learning models and returns their R2 scores'''
    try:

        r2_scores = {}
        for model_name, model in models.items():
            para = param.get(model_name, {})
            gs = GridSearchCV(model, para, cv=5, n_jobs=-1)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            r2_scores[model_name] = r2_score(y_test, y_test_pred)
        return r2_scores
    except Exception as e:
        raise CustomException(e, sys)
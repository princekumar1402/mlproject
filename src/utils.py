import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    '''Saves a Python object to a file using pickle'''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''Evaluates multiple machine learning models and returns their R2 scores'''
    try:

        r2_scores = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_score_value = r2_score(y_test, y_pred)
            r2_scores[model_name] = r2_score_value

        return r2_scores
    except Exception as e:
        raise CustomException(e, sys)
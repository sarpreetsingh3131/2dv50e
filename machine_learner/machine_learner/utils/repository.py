"""This file can load and save learning models from/to the hard drive."""
import traceback
import os
from sklearn.externals import joblib


def get(model, model_name, dir_path):
    try:
        pathFile = os.path.join(dir_path, model_name + '.pkl')
        if os.path.isfile(pathFile):
            model = joblib.load(pathFile)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
    return model


def create(model, model_name, dir_path):
    try:
        joblib.dump(model, dir_path + model_name + '.pkl')
    except Exception as e:
        traceback.print_tb(e.__traceback__)

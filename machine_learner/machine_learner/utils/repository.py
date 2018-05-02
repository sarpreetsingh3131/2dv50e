import traceback
from sklearn.externals import joblib

def get(model, model_name, dir_path):
    try:
        model = joblib.load(dir_path + model_name + '.pkl')
    except Exception as e:
        traceback.print_tb(e.__traceback__)
    return model


def create(model, model_name, dir_path):
    try:
        joblib.dump(model, dir_path + model_name + '.pkl')
    except Exception as e:
        traceback.print_tb(e.__traceback__)

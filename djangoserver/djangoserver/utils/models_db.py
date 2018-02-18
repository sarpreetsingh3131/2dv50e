from sklearn.externals import joblib
import os.path


def save_model(model, path):
    joblib.dump(model, path)


def get_saved_model(model, path):
    if os.path.isfile(path):
        print('model found')
        return joblib.load(path)
    print('model not found')
    return model

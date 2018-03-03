from sklearn.externals import joblib


def load_model(model, model_name, dir_path):
    try:
        model = joblib.load(dir_path + model_name + '.pkl')
        print(model_name, 'model found')
    except Exception:
        print(model_name, 'model not found')

    return model


def save_model(model, model_name, dir_path):
    try:
        joblib.dump(model, dir_path + model_name + '.pkl')
        print(model_name, 'model saved')
    except Exception:
        print(model_name, 'model not saved')

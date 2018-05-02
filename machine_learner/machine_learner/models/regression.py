import traceback
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDRegressor
from machine_learner.utils import repository

DIR_PATH = 'machine_learner/trained_models/regression/'

def training(features, target):
    try:
        model = repository.get(SGDRegressor(loss='epsilon_insensitive'), SGDRegressor.__name__, DIR_PATH)
        scaler = repository.get(MaxAbsScaler(), MaxAbsScaler.__name__, DIR_PATH)
        scaler.partial_fit(features)
        features = scaler.transform(features)
        model.partial_fit(features, target)
        repository.create(model, SGDRegressor.__name__, DIR_PATH)
        repository.create(scaler, MaxAbsScaler.__name__, DIR_PATH)
        return {'message': 'training successful '}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features):
    try:
        model = repository.get(SGDRegressor(loss='epsilon_insensitive'), SGDRegressor.__name__, DIR_PATH)
        scaler = repository.get(MaxAbsScaler(), MaxAbsScaler.__name__, DIR_PATH)
        features = scaler.transform(features)
        predictions = model.predict(features)
        response = {'predictions': [], 'adaptation_space': 0}
        for prediction in predictions:
            response['predictions'].append(float(prediction))
            if prediction < 10.0:
                response['adaptation_space'] += 1
        return response
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}
        
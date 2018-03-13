import traceback
from machine_learning.model_persistence.model_persistence import load_model, save_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveRegressor

DIR_PATH = 'machine_learning/trained_models/regression/test/'


def train(features, target):
    try:
        regressor = load_model(PassiveAggressiveRegressor(), PassiveAggressiveRegressor.__name__, DIR_PATH)
        scaler = load_model(StandardScaler(), StandardScaler.__name__, DIR_PATH)
        scaler.partial_fit(features)
        features = scaler.transform(features)
        regressor.partial_fit(features, target)
        save_model(regressor, PassiveAggressiveRegressor.__name__, DIR_PATH)
        save_model(scaler, StandardScaler.__name__, DIR_PATH)
        return {'message': 'trained successfully'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def test(features):
    try:
        regressor = load_model(PassiveAggressiveRegressor(), PassiveAggressiveRegressor.__name__, DIR_PATH)
        scaler = load_model(StandardScaler(), StandardScaler.__name__, DIR_PATH)
        features = scaler.transform(features)
        predictions = regressor.predict(features)
        response = {'result': [], 'adaptation_space': 0}

        for prediction in predictions:
            response['result'].append(float(prediction))
            if prediction < 10.0:
                response['adaptation_space'] += 1

        return response
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'testing failed'}

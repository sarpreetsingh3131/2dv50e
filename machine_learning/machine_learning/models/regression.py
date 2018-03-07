import traceback
from machine_learning.model_persistence.model_persistence import load_model, save_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor

DIR_PATH = 'machine_learning/trained_models/regression/'


def train(features, target):
    try:
        regressor = load_model(SGDRegressor(average=True), SGDRegressor.__name__, DIR_PATH)
        scaler = load_model(MinMaxScaler(), MinMaxScaler.__name__, DIR_PATH)
        scaler.partial_fit(features)
        features = scaler.transform(features)
        regressor.partial_fit(features, target)
        save_model(regressor, SGDRegressor.__name__, DIR_PATH)
        save_model(scaler, MinMaxScaler.__name__, DIR_PATH)
        return {'message': 'trained successfully'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def test(features):
    try:
        regressor = load_model(None, SGDRegressor.__name__, DIR_PATH)
        scaler = load_model(None, MinMaxScaler.__name__, DIR_PATH)
        features = scaler.transform(features)
        predictions = regressor.predict(features)
        response = {'result': [], 'adaptation_space': 0}
        
        for prediction in predictions:
            response['result'].append(float(prediction))  # todo here
            if float(prediction) < 10.0:
                response['adaptation_space'] += 1

        return response
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'testing failed'}

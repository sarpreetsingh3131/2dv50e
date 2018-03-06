import traceback
from machine_learning.model_persistence.model_persistence import load_model, save_model
from sklearn.preprocessing import MinMaxScaler

DIR_PATH = 'machine_learning/trained_models/regression/'


def training(features, target, model):
    try:
        regression = load_model(model(average=True), model.__name__, DIR_PATH)
        scaler = load_model(MinMaxScaler(), MinMaxScaler.__name__, DIR_PATH)
        scaler.partial_fit(features)
        features = scaler.transform(features)
        regression.partial_fit(features, target)
        save_model(regression, model.__name__, DIR_PATH)
        save_model(scaler, MinMaxScaler.__name__, DIR_PATH)
        return {'message': 'trained successfully'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features, model):
    try:
        regression = load_model(None, model.__name__, DIR_PATH)
        scaler = load_model(None, MinMaxScaler.__name__, DIR_PATH)
        features = scaler.transform(features)
        predictions = regression.predict(features)
        save_model(regression, model.__name__, DIR_PATH)
        response = {'result': [], 'adaptation_space': 0}

        for prediction in predictions:
            response['result'].append(float(prediction))  # todo here
            if float(prediction) < 0.10:
                response['adaptation_space'] += 1

        return response
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'testing failed'}

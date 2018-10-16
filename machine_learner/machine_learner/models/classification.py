import traceback
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from machine_learner.utils import repository

DIR_PATH = 'machine_learner/trained_models/classification/'


def training(features, target):
    try:
        model = repository.get(SGDClassifier(
            loss='hinge', penalty='l1'), SGDClassifier.__name__, DIR_PATH)
        scaler = repository.get(
            MinMaxScaler(), MinMaxScaler.__name__, DIR_PATH)
        scaler.partial_fit(features)
        features = scaler.transform(features)
        model.partial_fit(features, target, classes=np.array([0, 1]))
        repository.create(model, SGDClassifier.__name__, DIR_PATH)
        repository.create(scaler, MinMaxScaler.__name__, DIR_PATH)
        return {'message': 'training successful'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features):
    try:
        model = repository.get(SGDClassifier(
            loss='hinge', penalty='l1'), SGDClassifier.__name__, DIR_PATH)
        scaler = repository.get(
            MinMaxScaler(), MinMaxScaler.__name__, DIR_PATH)
        features = scaler.transform(features)
        predictions = model.predict(features)
        response = {'predictions': [], 'adaptation_space': 0}
        for prediction in predictions:
            response['predictions'].append(int(prediction))
            if prediction == 1:
                response['adaptation_space'] += 1
        return response
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}

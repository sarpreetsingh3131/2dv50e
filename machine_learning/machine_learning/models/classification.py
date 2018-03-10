import traceback
from machine_learning.model_persistence.model_persistence import load_model, save_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import numpy as np

DIR_PATH = 'machine_learning/trained_models/classification/'


def train(features, target):
    try:
        classifier = load_model(SGDClassifier(), SGDClassifier.__name__, DIR_PATH)
        scaler = load_model(StandardScaler(), StandardScaler.__name__, DIR_PATH)
        scaler.partial_fit(features)
        features = scaler.transform(features)
        classifier.partial_fit(features, target, classes=np.array([0, 1]))
        save_model(classifier, SGDClassifier.__name__, DIR_PATH)
        save_model(scaler, StandardScaler.__name__, DIR_PATH)
        return {'message': 'trained successfully'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def test(features):
    try:
        classifier = load_model(SGDClassifier(), SGDClassifier.__name__, DIR_PATH)
        scaler = load_model(StandardScaler(), StandardScaler.__name__, DIR_PATH)
        features = scaler.transform(features)
        predictions = classifier.predict(features)
        response = {'result': [], 'adaptation_space': 0}

        for prediction in predictions:
            response['result'].append(int(prediction))
            if prediction == 1:
                response['adaptation_space'] += 1

        return response
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'testing failed'}

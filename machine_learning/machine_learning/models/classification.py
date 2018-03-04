import traceback
from sklearn.model_selection import train_test_split
from machine_learning.model_persistence.model_persistence import load_model, save_model
import json

DIR_PATH = 'machine_learning/trained_models/classification/'


def training(features, target, model):
    try:
        classification = load_model(model(), model.__name__, DIR_PATH)
        classification.partial_fit(features, target, [0, 1])
        save_model(classification, model.__name__, DIR_PATH)
        return {'message': 'trained successfully'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features, model):
    try:
        classification = load_model(None, model.__name__, DIR_PATH)
        predictions = classification.predict(features)
        save_model(classification, model.__name__, DIR_PATH)
        response = {'result': [], 'adaptation_space': 0}

        for prediction in predictions:
            response['result'].append(int(prediction))
            if int(prediction) == 1:
                response['adaptation_space'] += 1  # packet loss less than 10
        return response
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'testing failed'}


def accuracy(features, target, model):
    try:
        training_features, testing_features, training_target, testing_target = train_test_split(
            features, target, test_size=0.3)
        classification = model()
        classification.partial_fit(training_features, training_target, [0, 1])
        score = classification.score(testing_features, testing_target)
        return {'accuracy': "{0:.2f}".format(score)}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'accuracy failed'}


"""
TRAINING DATA LENGTH = 816048
"""

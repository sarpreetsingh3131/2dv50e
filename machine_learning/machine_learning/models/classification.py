import traceback
from sklearn.model_selection import train_test_split
from machine_learning.model_persistence import model_persistence
from sklearn.preprocessing import StandardScaler
import json

DIR_PATH = 'machine_learning/trained_models/classification/'


def training(features, target, model):
    try:
        model_name = model.__name__
        model = model_persistence.load_model(model(), model_name, DIR_PATH)
        scaler = model_persistence.load_model(StandardScaler(), StandardScaler.__name__, DIR_PATH)
        scaler.fit(features)
        features = scaler.transform(features)
        model.partial_fit(features, target, [0, 1])
        model_persistence.save_model(model, model_name, DIR_PATH)
        model_persistence.save_model(scaler, StandardScaler.__name__, DIR_PATH)
        return {'message': 'trained successfully'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features, model):
    try:
        model_name = model.__name__
        model = model_persistence.load_model(model, model.__name__, DIR_PATH)
        scaler = model_persistence.load_model(StandardScaler(), StandardScaler.__name__, DIR_PATH)
        features = scaler.transform(features)
        predictions = model.predict(features)
        model_persistence.save_model(model, model_name, DIR_PATH)
        model_persistence.save_model(scaler, StandardScaler.__name__, DIR_PATH)
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

        scaler = StandardScaler()
        scaler.fit(training_features)
        training_features = scaler.transform(training_features)
        testing_features = scaler.transform(testing_features)

        model = model()
        model.partial_fit(training_features, training_target, [0, 1])
        score = "{0:.2f}".format(model.score(testing_features, testing_target))
        return {'accuracy': score}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'accuracy failed'}


"""
DEFAULT SETTINGS (LOG = HINGE, PENALTY = L2) ACCURACY RESULT
1) FIT = 0.95
2) SCALER + FIT = 0.95
3) PARTIAL FIT = 0.95
4) SCALER + PARTIAL FIT = 0.96

# LOSS FUNCTIONS
WHEN LOSS = LOG -> ACCURACY = 0.96
WHEN LOSS = MODIFIED_HUBER -> ACCURACY = 0.94
WHEN LOSS = SQUARED_HINGE -> ACCURACY = 0.95
WHEN LOSS = PERCEPTRON -> ACCURACY = 0.95
WHEN LOSS = SQUARED_LOSS -> ACCURACY = 0.51
WHEN LOSS = HUBER -> ACCURACY = 0.94
WHEN LOSS = EPSILON_INSENSITIVE -> ACCURACY = 0.94
WHEN LOSS = SQUARED_EPSILON_INSENSITIVE -> ACCURACY = 0.66

# PENALTY
DEFAULT SETTING WITH PENALTY = NONE -> ACCURACY = 0.95
DEFAULT SETTING WITH PENALTY = L1 -> ACCURACY = 0.96
DEFAULT SETTING WITH PENALTY = ELASTICNET -> ACCURACY = 0.96

"""

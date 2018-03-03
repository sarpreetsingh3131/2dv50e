import traceback
from sklearn.model_selection import train_test_split
from machine_learning.model_persistence import model_persistence
from sklearn.preprocessing import StandardScaler
import json

DIR_PATH = 'machine_learning/trained_models/regression/'


def training(features, target, model):
    try:
        model_name = model.__name__
        model = model_persistence.load_model(model(), model_name, DIR_PATH)
        scaler = model_persistence.load_model(StandardScaler(), StandardScaler.__name__, DIR_PATH)
        scaler.fit(features)
        features = scaler.transform(features)
        model.partial_fit(features, target)
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
            response['result'].append(float(prediction))
            if float(prediction) < 0.10:
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
        model.partial_fit(training_features, training_target)
        score = "{0:.2f}".format(model.score(testing_features, testing_target))
        return {'accuracy': score}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'accuracy failed'}

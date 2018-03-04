import traceback
from sklearn.model_selection import train_test_split
from machine_learning.model_persistence.model_persistence import load_model, save_model

DIR_PATH = 'machine_learning/trained_models/regression/'


def training(features, target, model):
    try:
        regression = load_model(model(average=True), model.__name__, DIR_PATH)
        regression.partial_fit(features, target)
        save_model(regression, model.__name__, DIR_PATH)
        return {'message': 'trained successfully'}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features, model):
    try:
        regression = load_model(None, model.__name__, DIR_PATH)
        predictions = regression.predict(features)
        save_model(regression, model.__name__, DIR_PATH)
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
        regression = model()
        regression.fit(training_features, training_target)
        score = regression.score(testing_features, testing_target)
        return {'accuracy': "{0:.2f}".format(score)}
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'accuracy failed'}

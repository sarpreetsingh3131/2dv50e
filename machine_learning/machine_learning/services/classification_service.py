from machine_learning.models import classification
from machine_learning.parser import testing_data, training_data
from sklearn.linear_model import SGDClassifier

MODELS = {
    'sgd_classifier': SGDClassifier
}


def run(mode, model_name, json_testing_data):
    if mode == 'training':
        data = training_data.parse(type='classification')
        return classification.training(data['features'], data['target'], MODELS[model_name])

    elif mode == 'testing':
        data = testing_data.parse(json_testing_data)
        return classification.testing(data['features'], MODELS[model_name])

    elif mode == 'accuracy':
        data = training_data.parse(type='classification')
        return classification.accuracy(data['features'], data['target'], MODELS[model_name])

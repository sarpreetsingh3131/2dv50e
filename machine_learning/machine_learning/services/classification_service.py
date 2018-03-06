from machine_learning.models import classification
from machine_learning.parser.parser import parse_data
from sklearn.linear_model import SGDClassifier

MODELS = {
    'sgd_classifier': SGDClassifier
}


def run(mode, model_name, data):
    if mode == 'training':
        features, target = parse_data(data, target_type='classification')
        return classification.training(features, target, MODELS[model_name])

    elif mode == 'testing':
        features, target = parse_data(data)
        return classification.testing(features, MODELS[model_name])

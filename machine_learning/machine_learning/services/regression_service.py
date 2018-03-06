from machine_learning.models import regression
from machine_learning.parser.parser import parse_data
from sklearn.linear_model import SGDRegressor


MODELS = {
    'sgd_regressor': SGDRegressor
}


def run(mode, model_name, data):
    if mode == 'training':
        features, target = parse_data(data, target_type='regression')
        return regression.training(features, target, MODELS[model_name])

    elif mode == 'testing':
        features, target = parse_data(data)
        return regression.testing(features, MODELS[model_name])

from machine_learning.models import regression
from machine_learning.parser import testing_data, training_data
from sklearn.linear_model import SGDRegressor


MODELS = {
    'sgd_regressor': SGDRegressor
}


def run(mode, model_name, json_testing_data):
    if mode == 'training':
        data = training_data.parse(type='regression')
        return regression.training(data['features'], data['target'], MODELS[model_name])

    elif mode == 'testing':
        data = testing_data.parse(json_testing_data)
        return regression.testing(data['features'], MODELS[model_name])

    elif mode == 'accuracy':
        data = training_data.parse(type='classification')
        return regression.accuracy(data['features'], data['target'], MODELS[model_name])

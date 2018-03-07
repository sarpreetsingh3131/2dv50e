from machine_learning.models import regression
from machine_learning.parser.parser import parse_data


def run(mode, data):
    if mode == 'training':
        features, target = parse_data(data, target_type='regression')
        return regression.train(features, target)

    elif mode == 'testing':
        features, target = parse_data(data)
        return regression.test(features)

    else:
        return {'message': 'invalid mode'}

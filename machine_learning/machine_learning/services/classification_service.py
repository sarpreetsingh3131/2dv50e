from machine_learning.models import classification
from machine_learning.parser.parser import parse_data


def run(mode, data):
    if mode == 'training':
        features, target = parse_data(data, target_type='classification')
        return classification.train(features, target)

    elif mode == 'testing':
        features, target = parse_data(data)
        return classification.test(features)

    else:
        return {'message': 'invalid mode'}

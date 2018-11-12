"""This file contains training and testing function with regression

A lot of work is to be done here too.
"""

import traceback
import os
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDRegressor
from machine_learner.utils import repository

# Path to the directory where the models are saved
DIR_PATH = os.path.join('machine_learner', 'trained_models', 'regression')



# Als adapt to multiple goals, by different models
# or maybe there is some other type of multiple goal model you can find
# you would have to auto model select for every model
def training(features, target, cycle):
    try:
        if cycle != 1:
            model = repository.get(SGDRegressor.__name__, DIR_PATH)
            scaler = repository.get(MaxAbsScaler.__name__, DIR_PATH)
        else:
            model = SGDRegressor(loss='epsilon_insensitive', penalty='l2')
            scaler = MaxAbsScaler()

        scaler.partial_fit(features)

        features = scaler.transform(features)

        model.partial_fit(features, target)

        repository.create(model, SGDRegressor.__name__, DIR_PATH)
        repository.create(scaler, MaxAbsScaler.__name__, DIR_PATH)

        return {'message': 'training successful '}

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features):
    try:
        model = repository.get(SGDRegressor.__name__, DIR_PATH)
        scaler = repository.get(MaxAbsScaler.__name__, DIR_PATH)

        features = scaler.transform(features)

        predictions = model.predict(features)

        response = {'predictions': [], 'adaptation_space': 0}

        for prediction in predictions:

            response['predictions'].append(float(prediction))

            if prediction < 10.0:
                response['adaptation_space'] += 1

        return response
        
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}

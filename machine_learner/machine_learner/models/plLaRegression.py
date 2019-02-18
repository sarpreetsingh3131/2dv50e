import traceback
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from machine_learner.utils import repository

DIR_PATH = os.path.join('machine_learner', 'trained_models', 'pllaregression')

def training(features, target_pl, target_la, cycle):

    if cycle != 1:
        model_pl = repository.get(SGDRegressor.__name__ + '_pl', DIR_PATH)
        scaler_pl = repository.get(MinMaxScaler.__name__ + '_pl', DIR_PATH)
        model_la = repository.get(SGDRegressor.__name__ + '_la', DIR_PATH)
        scaler_la = repository.get(MinMaxScaler.__name__ + '_la', DIR_PATH)
    else:
        model_pl = SGDRegressor()
        scaler_pl = MinMaxScaler()
        model_la = SGDRegressor()
        scaler_la = MinMaxScaler()

    scaler_pl.partial_fit(features)
    scaler_la.partial_fit(features)
    features_pl = scaler_pl.transform(features)
    features_la = scaler_la.transform(features)

    model_pl.partial_fit(features_pl, target_pl)
    model_la.partial_fit(features_la, target_la)

    repository.create(model_pl, SGDRegressor.__name__ + '_pl', DIR_PATH)
    repository.create(model_la, SGDRegressor.__name__ + '_la', DIR_PATH)
    repository.create(scaler_pl, MinMaxScaler.__name__ + '_pl', DIR_PATH)
    repository.create(scaler_la, MinMaxScaler.__name__ + '_la', DIR_PATH)

    return {'message': 'training successful'}
    


def testing(features):

    try:
    
        model_pl = repository.get(SGDRegressor.__name__ + '_pl', DIR_PATH)
        scaler_pl = repository.get(MinMaxScaler.__name__ + '_pl', DIR_PATH)
        model_la = repository.get(SGDRegressor.__name__ + '_la', DIR_PATH)
        scaler_la = repository.get(MinMaxScaler.__name__ + '_la', DIR_PATH)

        features_pl = scaler_pl.transform(features)
        features_la = scaler_la.transform(features)

        predictions_pl = model_pl.predict(features_pl)
        predictions_la = model_la.predict(features_la)

        response = {'predictions_pl': [], 'predictions_la': [], 'adaptation_space': 0, 'adaptation_class': 'noGoals'}

        bothGoals = 0
        oneGoal = 0

        for prediction_pl, prediction_la in zip(predictions_pl, predictions_la):

            response['predictions_pl'].append(float(prediction_pl))
            response['predictions_la'].append(float(prediction_la))
            
            # TODO centralize goals for the python module as well 
            # (currently only centralized in the SMC config properties file in the java module)
            prediction = (1 if prediction_pl < 10 else 0) + (2 if prediction_la < 5 else 0)

            if prediction == 3:
                bothGoals += 1
            elif prediction == 2 or prediction == 1:
                oneGoal += 1
        
        if bothGoals == 0:
            response['adaptation_space'] = oneGoal
            response['adaptation_class'] = 'oneGoal'
        else:
            response['adaptation_space'] = bothGoals
            response['adaptation_class'] = 'twoGoals'
        
        return response

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'testing failed'}

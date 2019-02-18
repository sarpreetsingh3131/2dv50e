import traceback
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from machine_learner.utils import repository

DIR_PATH = os.path.join('machine_learner', 'trained_models', 'pllaclassification')

def training(features, target, cycle):
    try:
        if cycle != 1:
            model = repository.get(SGDClassifier.__name__, DIR_PATH)
            scaler = repository.get(MinMaxScaler.__name__, DIR_PATH)
        else:
            model = SGDClassifier(loss='hinge', penalty='l1')
            scaler = MinMaxScaler()


        scaler.partial_fit(features)
        features = scaler.transform(features)

        # online learn the adaptation options (features) and their targets (goal satisfactions)
        model.partial_fit(features, target, classes=np.array([0, 1, 2, 3]))

        # Save the scaler and classifier model
        repository.create(model, SGDClassifier.__name__, DIR_PATH)
        repository.create(scaler, MinMaxScaler.__name__, DIR_PATH)

        return {'message': 'training successful'}
    
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features):

    try:
    
        model = repository.get(SGDClassifier.__name__, DIR_PATH)
        scaler = repository.get(MinMaxScaler.__name__, DIR_PATH)

        features = scaler.transform(features)

        predictions = model.predict(features)

        response = {'predictions': [], 'adaptation_space': 0, 'adaptation_class': 'noGoals'}

        bothGoals = 0
        oneGoal = 0

        for prediction in predictions:
            response['predictions'].append(int(prediction))
            
            if prediction == 3:
                bothGoals += 1
            elif prediction == 2 or prediction == 1:
                oneGoal += 1
                
                
        if bothGoals == 0:
            # If no adaptation option is predicted to be of class 3,
            # consider all options which are predicted to meet a single goal
            response['adaptation_space'] = oneGoal
            response['adaptation_class'] = 'oneGoal'
        else:
            response['adaptation_space'] = bothGoals
            response['adaptation_class'] = 'twoGoals'
        
        return response

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}

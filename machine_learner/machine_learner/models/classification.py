import traceback
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.kernel_approximation import RBFSampler
from machine_learner.utils import repository

# Path to the directory where the models are saved
DIR_PATH = os.path.join('machine_learner', 'trained_models', 'classification')

def training(features, target, cycle):
    try:
        if cycle != 1:
            model = repository.get(PassiveAggressiveClassifier.__name__, DIR_PATH)
            scaler = repository.get(StandardScaler.__name__, DIR_PATH)
        else:
            model = PassiveAggressiveClassifier(loss='squared_hinge')
            scaler = StandardScaler()

        # add the new data to the existing scaler model, because the scaling depends on the data
        scaler.partial_fit(features)

        # scale the new features
        features = scaler.transform(features)

        # Perform online learning with the new features
        model.partial_fit(features, target, classes=np.array([0, 1]))

        # Store the model and scaler to respective files for later use
        repository.create(model, PassiveAggressiveClassifier.__name__, DIR_PATH)
        repository.create(scaler, StandardScaler.__name__, DIR_PATH)

        return {'message': 'training successful'}
    
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features):
    try:
        model = repository.get(PassiveAggressiveClassifier.__name__, DIR_PATH)
        scaler = repository.get(StandardScaler.__name__, DIR_PATH)
        
        features = scaler.transform(features)

        predictions = model.predict(features)

        response = {'predictions': [], 'adaptation_space': 0}


        for prediction in predictions:
            response['predictions'].append(int(prediction))
            if prediction == 1:
                response['adaptation_space'] += 1
        
        return response

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}

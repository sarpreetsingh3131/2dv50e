"""This file does the classification learning

There is a lot to be done here
"""

import traceback
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.kernel_approximation import RBFSampler
from machine_learner.utils import repository

# Path to the directory where the models are saved
DIR_PATH = os.path.join('machine_learner', 'trained_models', 'classification')

# TODO: addapt to a flexible model and multiple goals
# The goals will have to be added and synchronized between 
# this and activforms
# however, I do not think much has to be done here except the multiclass
# methode to choose
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
        
        # scale the features appropriatly
        # TODO: why don't you online learn the scaler here?
        # As far as I remember you do not need the prediction about them to scale them
        # contrary to the classifier
        # I could be wrong however.
        features = scaler.transform(features)

        # Make the prediction about the model
        predictions = model.predict(features)

        # Initialise the response with a prediction key 
        # pointing to an empty list
        # and an adaptation_space as 0 
        # which will be number of predictions that satifies the goals
        # TODO: if you have time left, change the space name
        # to something like size or something.
        # you will also have to change it in the SMCConnector
        response = {'predictions': [], 'adaptation_space': 0}


        for prediction in predictions:

            # TODO: why would you send all prediction back if you only look 
            # at the one that satisfy the goals in  the connector?
            response['predictions'].append(int(prediction))
            
            # TODO: change this 
            # if this happens in multiclass, add 1 if the predicted class is 0
            # because then the model checker will run on all of the adaption options
            if prediction == 1:
                response['adaptation_space'] += 1
        
        return response

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}

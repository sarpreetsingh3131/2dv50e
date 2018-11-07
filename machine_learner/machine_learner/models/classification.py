"""This file does the classification learning

There is a lot to be done here
"""

import traceback
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from machine_learner.utils import repository

#DIR_PATH = 'machine_learner/trained_models/classification/'
#changed the above path to be platform independent
DIR_PATH = os.path.join('machine_learner', 'trained_models', 'classification')

#TODO: addapt to a flexible model and multiple goals
# The goals will have to be added and synchronized between 
# this and activforms
# however, I do not think much has to be done here except the multiclass
# methode to choose
def training(features, target):

    try:

        # he makes this model and replaces it if he is able
        # to load the previous model
        # If not, he returns the classifier given to him 
        # This is an extremely lazy way of 
        # coding: if file doesn't exist, make one
        # This is inneficient
        # You make the model every time, every cycle...
        # Remove this later, same story for the scaler

        #The SGDClassifier.__name__ just returns a string
        # "SGDClassifier" and I presume it does the same 
        # equivalent thing for the other names

        #TODO: make get return null in case there is an error/no file and do here
        # if null then make one

        # also, you could do the automated model selection here
        # if it return null

        model = repository.get(SGDClassifier(
            loss='hinge', penalty='l1'), SGDClassifier.__name__, DIR_PATH)

        # same for the scaler
        scaler = repository.get(
            MinMaxScaler(), MinMaxScaler.__name__, DIR_PATH)

        # add the knew data to the existing scaler model, because the scaling depends on the data
        scaler.partial_fit(features)

        # scale the new features
        features = scaler.transform(features)

        # "online learn" from the new features
        model.partial_fit(features, target, classes=np.array([0, 1]))

        # You write the model and scaler back to the disk
        # with their new additions
        repository.create(model, SGDClassifier.__name__, DIR_PATH)
        repository.create(scaler, MinMaxScaler.__name__, DIR_PATH)

        return {'message': 'training successful'}
    
    except Exception as e:
        
        #something went terribly wrong
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features):

    try:
    
        # The same problem with get as above but even worse
        # Here he would test on a completely new model
        # if non found. It's @#$! an should be dealt with
        model = repository.get(SGDClassifier(
            loss='hinge', penalty='l1'), SGDClassifier.__name__, DIR_PATH)

        # same problem as above
        scaler = repository.get(
            MinMaxScaler(), MinMaxScaler.__name__, DIR_PATH)
        
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
            if prediction == 1:
                response['adaptation_space'] += 1
        
        return response

    # something went seriously wrong
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}

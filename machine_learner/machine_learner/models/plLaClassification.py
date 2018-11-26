import traceback
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from machine_learner.utils import repository

DIR_PATH = os.path.join('machine_learner', 'trained_models', 'plLaClassification')

#TODO: addapt to a flexible model and multiple goals
# The goals will have to be added and synchronized between 
# this and activforms
# however, I do not think much has to be done here except the multiclass
# methode to choose
def training(features, target, cycle):
    try:

        # also, you could do the automated model selection here
        # if it return null

        # TODO: change to selected model
        # n_jobs is the amout of CPUs to use.
        # you should change it to 4 if your pc can handle it
        # SGDClassifier uses OVA.
        modelName = SGDClassifier.__name__
        if cycle != 1:
            model = repository.get(SGDClassifier.__name__, DIR_PATH)
            scaler = repository.get(MinMaxScaler.__name__, DIR_PATH)
        else:
            model = SGDClassifier(loss='hinge', penalty='l1')
            scaler = MinMaxScaler()

        #TODO: make dimensionality reduction here.

        # add the knew data to the existing scaler model, because the scaling depends on the data
        scaler.partial_fit(features)

        # scale the new features
        features = scaler.transform(features)

        # "online learn" from the new features
        model.partial_fit(features, target, classes=np.array([0, 1, 2, 3]))

        # You write the model and scaler back to the disk
        # with their new additions
        # TODO: write dimensionality reduction away
        repository.create(model, SGDClassifier.__name__, DIR_PATH)
        repository.create(scaler, MinMaxScaler.__name__, DIR_PATH)

        return {'message': 'training successful'}
    
    except Exception as e:
        
        #something went terribly wrong
        traceback.print_tb(e.__traceback__)
        return {'message': 'training failed '}


def testing(features):

    try:
    
        model = repository.get(SGDClassifier.__name__, DIR_PATH)
        scaler = repository.get(MinMaxScaler.__name__, DIR_PATH)

        # TODO: dimensionality reduction
        
        # scale the features appropriatly
        # TODO: why don't you online learn the scaler here?
        # As far as I remember you do not need the prediction about them to scale them
        # contrary to the classifier
        # I could be wrong however.
        # I think because you get a return training round after each testing round
        # from activforms and the scaler would be fitted twice on the same data
        features = scaler.transform(features)

        # Make the prediction about the model
        predictions = model.predict(features)

        # Initialise the response with a prediction key 
        # pointing to an empty list
        # and an adaptation_space as 0 
        # which will be number of predictions that satifies the goals
        # if you have time left, change the space name
        # to something like size or something.
        # you will also have to change it in the SMCConnector
        # Its no use, leave it as be.
        response = {'predictions': [], 'adaptation_space': 0}

        bothGoals = 0
        oneGoal = 0

        for prediction in predictions:

            # why would you send all prediction back if you only look 
            # at the one that satisfy the goals in  the connector?
            # You check again for all prediction in the connector...
            # Could be optimised but again, time can be spend better
            response['predictions'].append(int(prediction))
            
            
            # if this happens in multiclass, add 1 if the predicted class is 0
            # because then the model checker will run on all of the adaption options
            
            # if there is an adaptation that satisfies both goals, 
            # the positive number of the amount of options that do will be send back
            # if not and there are options which satisfy one goal, the negation of
            # the amount of goals will be send back
            # If not option satifies one or both goals, 0 is returned
            if prediction == 3:
                bothGoals += 1
            elif prediction == 2 or prediction == 1:
                oneGoal += 1
        
        if bothGoals == 0:
            response['adaptation_space'] = -oneGoal
        else:
            response['adaptation_space'] = bothGoals
        
        return response

    # something went seriously wrong
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return{'message': 'testing failed'}

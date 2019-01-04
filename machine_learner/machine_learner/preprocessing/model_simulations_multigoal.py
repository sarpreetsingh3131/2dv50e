import copy
import random
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, SGDRegressor, PassiveAggressiveRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


ADAP_SIZE = 729

scalers = [MinMaxScaler, StandardScaler, MaxAbsScaler, None]

penalty_sgd = ['l1', 'l2', 'elasticnet', 'none']
loss_sgd_regr = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
loss_sgd_class = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'] + loss_sgd_regr

loss_pa = ['hinge', 'squared_hinge']

penalty_per = ['l1', 'l2', 'elasticnet', None]


# Classifiers in format (model, [loss functions], [penalties], [scalers])
classifiers = [
    (PassiveAggressiveClassifier, loss_pa, [None], scalers),
    (SGDClassifier, loss_sgd_class, penalty_sgd, scalers),
    (Perceptron, [None], penalty_per, scalers),
    (GaussianNB, [None], [None], scalers),
    # (BernoulliNB, [None], [None], scalers)
    # (MultinomialNB, [None], [None], [None]),
]

regressors = [
    (SGDRegressor, ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], penalty_per[:3], scalers),
    (PassiveAggressiveRegressor, ['epsilon_insensitive', 'squared_epsilon_insensitive'], [None], scalers)
]



data = json.load(open(os.path.join('machine_learner','collected_data','dataset_selected_features_Classification.json')))
features_global = data['features']
targets_pl_class_global = data['target_classification_packetloss']
targets_la_class_global = data['target_classification_latency']
targets_pl_regr_global = data['target_regression_packetloss']
targets_la_regr_global = data['target_regression_latency']
targets_ec_regr_global = data['target_regression_energyconsumption']


class ModelEncapsClass:
    def __init__(self, model, loss, penalty, separate=False):
        self.separate = separate
        self.loss = loss
        self.penalty = penalty
        self.model = model
        self.models = []

        for _ in range(2 if separate else 1):
            if loss == None and penalty == None:
                m = model()
            elif loss == None and penalty != None:
                m = model(penalty=penalty)
            elif loss != None and penalty == None:
                m = model(loss=loss)
            else:
                m = model(loss=loss, penalty=penalty)
            
            self.models.append(m)
        

    def getName(self):
        '''
        Returns a tuple with 3 entries containing the model, loss and penalty names
        '''
        return (self.model.__name__ + ('-sep' if self.separate else ''), \
            self.loss.replace('_', '' ) if self.loss != None else 'None', \
            self.penalty if self.penalty != None else 'None')


    def predict(self, features):
        '''
        Proxy method
        '''

        if self.separate:
            # The first model predicts packet loss, the second latency
            pred_pl = self.models[0].predict(features)
            pred_la = self.models[1].predict(features)
            return [pred_pl[i] + (2*pred_la[i]) for i in range(len(pred_pl))]
        else:
            return self.models[0].predict(features).tolist()
    

    def partial_fit(self, features, targets_pl, targets_la):
        '''
        Proxy method 
        '''
        if self.separate:
            self.models[0].partial_fit(features, targets_pl, classes=np.array([0,1]))
            self.models[1].partial_fit(features, targets_la, classes=np.array([0,1]))
        else:
            comb_targets = [targets_pl[i] + (2*targets_la[i]) for i in range(len(targets_pl))]
            self.models[0].partial_fit(features, comb_targets, classes=np.array([0,1,2,3]))


class ModelEncapsRegr:
    def __init__(self, model, loss, penalty):
        self.loss = loss
        self.penalty = penalty
        self.model = model
        self.models = []

        for _ in range(2):
            if loss == None and penalty == None:
                m = model()
            elif loss == None and penalty != None:
                m = model(penalty=penalty)
            elif loss != None and penalty == None:
                m = model(loss=loss)
            else:
                m = model(loss=loss, penalty=penalty)
            
            self.models.append(m)

    def getName(self):
        '''
        Returns a tuple with 3 entries containing the model, loss and penalty names
        '''
        return (self.model.__name__ , \
            self.loss.replace('_', '' ) if self.loss != None else 'None', \
            self.penalty if self.penalty != None else 'None') 

    def predict(self, features):
        pred_pl = self.models[0].predict(features)
        pred_la = self.models[1].predict(features)

        class_pl = [1 if i < 10 else 0 for i in pred_pl]
        class_la = [1 if i < 5 else 0 for i in pred_la]

        return [class_pl[i] + (2 * class_la[i]) for i in range(len(features))]

    def partial_fit(self, features, targets_pl, targets_la):
        self.models[0].partial_fit(features, targets_pl)
        self.models[1].partial_fit(features, targets_la)



def generateResultingFileClass(model, scaler, loss, penalty, separate=False, amtTrainingCycles = 30):
    '''
    Generates an output file in the format of the output file from the MLAdjustment runmode for
    the provided model, scaler, loss and penalty.

    @param separate:    decides wether the predictions for all goals should be done separately by different 
                        binary classifiers or by one classifier with multiple (more than 2) classes
    '''

    start = time.process_time()
    data = []

    cycles = int(len(targets_pl_class_global) / ADAP_SIZE)
    model = ModelEncapsClass(model, loss, penalty, separate)
    model_name, loss_name, pen_name = model.getName()

    name = f'{model_name}_{loss_name + "." + pen_name}_{scaler.__name__ if scaler != None else "None"}'
    outputPath = os.path.join('machine_learner', 'collected_data', 'target', name + '.json')
    
    if scaler != None:
        scaler = scaler()

    # Simulate the classifier over all the cycles
    for i in range(cycles):
        # Extract the features and targets for the different goals
        features = copy.deepcopy(features_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE])
        targets_pl_class = targets_pl_class_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_la_class = targets_la_class_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_pl_regr = targets_pl_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_la_regr = targets_la_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_ec_regr = targets_ec_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        

        # The predictions by the classifier
        classBefore = []
        classAfter = []

        # Differentiate between training and testing cycles
        if i < amtTrainingCycles:
            if scaler != None:
                scaler.partial_fit(features)
                features = scaler.transform(features)

            if i==0: # Cannot make predictions at the first cycle since the classifier hasn't been trained yet
                classBefore = [-1 for i in range(ADAP_SIZE)]
            else:
                classBefore = model.predict(features)
            
            model.partial_fit(features, targets_pl_class, targets_la_class)
            classAfter = model.predict(features)

        else:
            if scaler != None:
                features = scaler.transform(features)
            classBefore = model.predict(features)

            # Determine the class(es) of predictions that should be used for online learning
            if 3 in classBefore:
                goals = [3]
            elif 1 in classBefore or 2 in classBefore:
                goals = [1,2]
            else:
                goals = [0]
            
            # Collect the samples for online learning
            indices = [i for i in range(ADAP_SIZE) if classBefore[i] in goals]
            model.partial_fit(np.array(features)[indices].tolist(), \
                np.array(targets_pl_class)[indices].tolist(), \
                np.array(targets_la_class)[indices].tolist())
            
            classAfter = model.predict(features)
    
        
        data.append({
            'cycle' : i+1,
            'training' : 'true' if i < amtTrainingCycles else 'false',
            'adaptationOptions': {
                'packetLoss' : targets_pl_regr,
                'energyConsumption' : targets_ec_regr,
                'latency' : targets_la_regr,
                'classificationBefore' : classBefore,
                'classificationAfter' : classAfter,
                'regressionPLBefore' : [-1] * ADAP_SIZE,
                'regressionPLAfter' : [-1] * ADAP_SIZE,
                'regressionLABefore' : [-1] * ADAP_SIZE,
                'regressionLAAfter' : [-1] * ADAP_SIZE
            }
        })

    with open(outputPath, 'w') as f:
        json.dump(data, f)

    end = time.process_time()
    print(f'{end-start:.2f} seconds:\t{name}')





def generateResultingFileRegr(model, scaler, loss, penalty, amtTrainingCycles = 30):
    '''
    This function treats the results of the regressor as predicted values from a classifier (class 0 - 3)
    '''
    # FIXME: if time left, store actuall regression results and perform regression metrics on these
    # TODO: merge with method above

    start = time.process_time()
    data = []

    cycles = int(len(targets_pl_class_global) / ADAP_SIZE)
    model = ModelEncapsRegr(model, loss, penalty)
    model_name, loss_name, pen_name = model.getName()

    name = f'{model_name}_{loss_name + "." + pen_name}_{scaler.__name__ if scaler != None else "None"}'
    outputPath = os.path.join('machine_learner', 'collected_data', 'target', name + '.json')
    
    if scaler != None:
        scaler = scaler()

    # Simulate the classifier over all the cycles
    for i in range(cycles):
        # Extract the features and targets for the different goals
        features = copy.deepcopy(features_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE])
        targets_pl_regr = targets_pl_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_la_regr = targets_la_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_ec_regr = targets_ec_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        

        # The predictions by the classifier
        classBefore = []
        classAfter = []

        # Differentiate between training and testing cycles
        if i < amtTrainingCycles:
            if scaler != None:
                scaler.partial_fit(features)
                features = scaler.transform(features)

            if i==0: # Cannot make predictions at the first cycle since the classifier hasn't been trained yet
                classBefore = [-1 for i in range(ADAP_SIZE)]
            else:
                classBefore = model.predict(features)
            
            model.partial_fit(features, targets_pl_regr, targets_la_regr)
            classAfter = model.predict(features)

        else:
            if scaler != None:
                features = scaler.transform(features)
            classBefore = model.predict(features)

            # Determine the class(es) of predictions that should be used for online learning
            if 3 in classBefore:
                goals = [3]
            elif 1 in classBefore or 2 in classBefore:
                goals = [1,2]
            else:
                goals = [0]
            
            # Collect the samples for online learning
            indices = [i for i in range(ADAP_SIZE) if classBefore[i] in goals]
            model.partial_fit(np.array(features)[indices].tolist(), \
                np.array(targets_pl_regr)[indices].tolist(), \
                np.array(targets_la_regr)[indices].tolist())
            
            classAfter = model.predict(features)
    
        data.append({
            'cycle' : i+1,
            'training' : 'true' if i < amtTrainingCycles else 'false',
            'adaptationOptions': {
                'packetLoss' : targets_pl_regr,
                'energyConsumption' : targets_ec_regr,
                'latency' : targets_la_regr,
                'classificationBefore' : classBefore,
                'classificationAfter' : classAfter,
                'regressionPLBefore' : [-1] * ADAP_SIZE,
                'regressionPLAfter' : [-1] * ADAP_SIZE,
                'regressionLABefore' : [-1] * ADAP_SIZE,
                'regressionLAAfter' : [-1] * ADAP_SIZE
            }
        })

    with open(outputPath, 'w') as f:
        json.dump(data, f)

    end = time.process_time()
    print(f'{end-start:.2f} seconds:\t{name}')





if __name__ == '__main__':
    for model, losses, penalties, scalers in classifiers:
        for scaler in scalers:
            for penalty in penalties:
                for loss in losses:
                    generateResultingFileClass(model, scaler, loss, penalty, False)


    for model, losses, penalties, scalers in regressors:
        for scaler in scalers:
            for penalty in penalties:
                for loss in losses:
                    generateResultingFileRegr(model, scaler, loss, penalty)


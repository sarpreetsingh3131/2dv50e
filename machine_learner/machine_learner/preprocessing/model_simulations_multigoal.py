import copy
from random import shuffle
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, SGDRegressor, PassiveAggressiveRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


# NOTE: adjust this when switching between the different networks
ADAP_SIZE = 216
# ADAP_SIZE = 4096
CAP_VERIFICATION_TIME = 570 * 1000

scalers = [MinMaxScaler, StandardScaler, MaxAbsScaler]

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
]

regressors = [
    (SGDRegressor, ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], penalty_per[:3], scalers),
    (PassiveAggressiveRegressor, ['epsilon_insensitive', 'squared_epsilon_insensitive'], [None], scalers)
]



def getSimpleModelName(originalName):
    if originalName == "PassiveAggressiveClassifier":
        return "PA-C"
    elif originalName == "PassiveAggressiveRegressor":
        return "PA-R"
    elif originalName == "SGDClassifier":
        return "SGD-C"
    elif originalName == "SGDRegressor":
        return "SGD-R"
    else:
        return originalName

data = json.load(open(os.path.join('machine_learner','collected_data','dataset_selected_features.json')))
features_global = data['features']
targets_pl_class_global = data['target_classification_packetloss']
targets_la_class_global = data['target_classification_latency']
targets_pl_regr_global = data['target_regression_packetloss']
targets_la_regr_global = data['target_regression_latency']
targets_ec_regr_global = data['target_regression_energyconsumption']
verification_times_global = data['verification_times']
del data


class ModelEncapsClass:
    def __init__(self, model, loss, penalty):
        self.loss = loss
        self.penalty = penalty
        self.model = model

        if loss == None and penalty == None:
            self.single_model = model()
        elif loss == None and penalty != None:
            self.single_model = model(penalty=penalty)
        elif loss != None and penalty == None:
            self.single_model = model(loss=loss)
        else:
            self.single_model = model(loss=loss, penalty=penalty)
        

    def getName(self):
        '''
        Returns a tuple with 3 entries containing the model, loss and penalty names
        '''
        return (getSimpleModelName(self.model.__name__) , \
            self.loss.replace('_', '' ) if self.loss != None else 'None', \
            self.penalty if self.penalty != None else 'None')


    def predict(self, features):
        '''
        Proxy method
        '''
        return self.single_model.predict(features).tolist()
    

    def partial_fit(self, features, targets_pl, targets_la):
        '''
        Proxy method 
        '''
        comb_targets = [targets_pl[i] + (2*targets_la[i]) for i in range(len(targets_pl))]
        self.single_model.partial_fit(features, comb_targets, classes=np.array([0,1,2,3]))



class ModelEncapsRegr:
    def __init__(self, models, losses, penalties):
        self.loss_pl, self.loss_la = losses
        self.penalty_pl, self.penalty_la = penalties
        self.model_pl, self.model_la = models
        self.models = []

        for i in range(2):
            if losses[i] == None and penalties[i] == None:
                m = models[i]()
            elif losses[i] == None and penalties[i] != None:
                m = models[i](penalty=penalties[i])
            elif losses[i] != None and penalties[i] == None:
                m = models[i](loss=losses[i])
            else:
                m = models[i](loss=losses[i], penalty=penalties[i])
            
            self.models.append(m)

    def getName(self):
        '''
        Returns a tuple with 3 entries containing the model, loss and penalty names
        '''
        return (getSimpleModelName(self.model_pl.__name__) + '.' + getSimpleModelName(self.model_la.__name__), \
            (self.loss_pl.replace('_', '' ) if self.loss_pl != None else 'None') + ('.' + self.loss_la.replace('_', '' ) if self.loss_la != None else 'None'), \
            (self.penalty_pl if self.penalty_pl != None else 'None') + '.' + (self.penalty_la if self.penalty_la != None else 'None')) 

    def predict(self, features_pl, features_la):
        pred_pl = self.models[0].predict(features_pl)
        pred_la = self.models[1].predict(features_la)

        class_pl = [1 if i < 10 else 0 for i in pred_pl]
        class_la = [1 if i < 5 else 0 for i in pred_la]

        return [class_pl[i] + (2 * class_la[i]) for i in range(len(features_pl))]

    def partial_fit(self, features_pl, features_la, targets_pl, targets_la):
        self.models[0].partial_fit(features_pl, targets_pl)
        self.models[1].partial_fit(features_la, targets_la)




def simulateModelWithTimes(model, scaler, loss, penalty, amtTrainingCycles = 30, explorationPercentage = 0.05):
    isClassifier = model in [i[0] for i in classifiers]

    start = time.perf_counter()
    data = []

    cycles = int(len(targets_pl_class_global) / ADAP_SIZE)
    if isClassifier:
        model = ModelEncapsClass(model, loss, penalty)
    else:
        model = ModelEncapsRegr([model]*2, [loss]*2, [penalty]*2)
    model_name, loss_name, pen_name = model.getName()
    # FIXME training cycles and exploration percentage temporarily stored here
    model_name += f'-{amtTrainingCycles}-{explorationPercentage}'

    name = f'{model_name}_{loss_name}_{pen_name}_{scaler.__name__ if scaler != None else "None"}'
    outputPath = os.path.join('machine_learner', 'collected_data', 'target', name + '.json')
    
    if scaler != None:
        scaler = scaler()

    # Index used to keep track of the last configuration that was learned in a single cycle
    learning_index = 0


    # Simulate the classifier over all the cycles
    for i in range(cycles):
        # Extract the features and targets for the different goals in this cycle
        features = getCopyFeatures(i*ADAP_SIZE,(i+1)*ADAP_SIZE)
        targets_pl_class = targets_pl_class_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_la_class = targets_la_class_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_pl_regr = targets_pl_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_la_regr = targets_la_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        targets_ec_regr = targets_ec_regr_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        verification_times = verification_times_global[i*ADAP_SIZE:(i+1)*ADAP_SIZE]
        

        # The predictions by the classifier (or regressor)
        classBefore = []
        classAfter = []

        verifiedOptions = [0 for i in range(ADAP_SIZE)]

        learning_time = 0
        verification_time_total = sum(verification_times)
        verification_time_with_learning = 0

        # Differentiate between training and testing cycles
        if i < amtTrainingCycles:

            # Limited training since we have a cap on verification times
            beginIndex = learning_index
            endIndex, verification_time_with_learning = \
                determineLearningSamples(verification_times, beginIndex, CAP_VERIFICATION_TIME)
                
            if beginIndex < endIndex:
                features_learning = features[beginIndex:endIndex]
                if isClassifier:
                    targets_learning_pl = targets_pl_class[beginIndex:endIndex]
                    targets_learning_la = targets_la_class[beginIndex:endIndex]
                else:
                    targets_learning_pl = targets_pl_regr[beginIndex:endIndex]
                    targets_learning_la = targets_la_regr[beginIndex:endIndex]

                for opIndex in range(beginIndex, endIndex):
                    verifiedOptions[opIndex] = 1
            else:
                features_learning = features[beginIndex:] + features[:endIndex]
                if isClassifier:
                    targets_learning_pl = targets_pl_class[beginIndex:] + targets_pl_class[:endIndex]
                    targets_learning_la = targets_la_class[beginIndex:] + targets_la_class[:endIndex]
                else:
                    targets_learning_pl = targets_pl_regr[beginIndex:] + targets_pl_regr[:endIndex]
                    targets_learning_la = targets_la_regr[beginIndex:] + targets_la_regr[:endIndex]

                for opIndex in range(beginIndex, ADAP_SIZE):
                    verifiedOptions[opIndex] = 1
                for opIndex in range(endIndex):
                    verifiedOptions[opIndex] = 1
                
            learning_index = endIndex

            # Actual training cycles
            startLearningTime = time.perf_counter()
            if scaler != None:
                scaler.partial_fit(features_learning)
                features_learning = scaler.transform(features_learning)

            classBefore = [-1 for i in range(ADAP_SIZE)]
            
            if isClassifier:
                model.partial_fit(features_learning, targets_learning_pl, targets_learning_la)
            else:
                model.partial_fit(features_learning, features_learning, targets_learning_pl, targets_learning_la)
                
            learning_time = (time.perf_counter() - startLearningTime) * 1000

            classAfter = [-1 for i in range(ADAP_SIZE)]


        else:
            # Testing cycle

            startLearningTime = time.perf_counter()

            if scaler != None:
                features = scaler.transform(features)

            if isClassifier:
                classBefore = model.predict(features)
            else:
                classBefore = model.predict(features, features)            

            meanTime = time.perf_counter() - startLearningTime
            # Determine the class(es) of predictions that should be used for online learning
            if 3 in classBefore:
                # If the options which are predicted to be of class 3 go over the verification time cap,
                # take a random sample of all the options with prediction 3.
                # Otherwise, take a percentage of options of class 1 and 2 as well (exploration)
                indices3 = [i for i in range(ADAP_SIZE) if classBefore[i] in [3]]
                verifTime3 = sum(np.array(verification_times)[indices3])
                if verifTime3 > CAP_VERIFICATION_TIME:
                    indices = sampleOfOptions(verification_times, indices3, CAP_VERIFICATION_TIME)
                else:
                    indices12 = [i for i in range(ADAP_SIZE) if classBefore[i] in [1,2]]
                    sample = sampleOfOptions(verification_times, indices12, CAP_VERIFICATION_TIME-verifTime3, explorationPercentage)
                    indices = indices3 + sample
                    
            elif (1 in classBefore) or (2 in classBefore):
                indices12 = [i for i in range(ADAP_SIZE) if classBefore[i] in [1,2]]
                indices = sampleOfOptions(verification_times, indices12, CAP_VERIFICATION_TIME)
            else:
                indices0123 = [i for i in range(ADAP_SIZE) if classBefore[i] in [0,1,2,3]]
                indices = sampleOfOptions(verification_times, indices0123, CAP_VERIFICATION_TIME)
            

            startLearningTime = time.perf_counter()
            # Collect the samples for online learning
            if isClassifier:
                model.partial_fit(np.array(features)[indices].tolist(), \
                    np.array(targets_pl_class)[indices].tolist(), \
                    np.array(targets_la_class)[indices].tolist())
            else:
                model.partial_fit(np.array(features)[indices].tolist(), \
                    np.array(features)[indices].tolist(), \
                    np.array(targets_pl_regr)[indices].tolist(), \
                    np.array(targets_la_regr)[indices].tolist())

            learning_time = (time.perf_counter() - startLearningTime + meanTime) * 1000

            if isClassifier:
                classAfter = model.predict(features)
            else:
                classAfter = model.predict(features, features)

            for index in indices:
                verifiedOptions[index] = 1

            # Only count the verification times of the configurations that were actually considered
            verification_time_with_learning = sum(np.array(verification_times)[indices].tolist())

        
        data.append({
            'cycle' : i+1,
            'learningTime' : learning_time,
            'verificationTimeWithLearning' : verification_time_with_learning,
            'verificationTimeWithoutLearning': verification_time_total,
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
                'regressionLAAfter' : [-1] * ADAP_SIZE,
                'verifiedConfigurations' : verifiedOptions
            }
        })

    with open(outputPath, 'w') as f:
        json.dump(data, f, indent=1)

    end = time.perf_counter()
    print(f'{end-start:.2f} seconds:\t{name}')


def determineLearningSamples(verifTimes, beginIndex, cappedTime):
    # Determines the resulting index, when starting from beginIndex, for all the options
    # which can be verified in time.
    # @param verifTimes: the verification times of all the options
    # @param beginIndex: the beginning index from which the options are considered (inclusive)
    # @param cappedTime: the limit on the amount of time for verification
    #
    # @returns endIndex, time: the ending index (exclusive), and the total verification times for the options in between
    endIndex = beginIndex
    time = 0

    while True:
        time += verifTimes[endIndex]
        endIndex = (endIndex+1) % len(verifTimes)

        if (time + verifTimes[endIndex] > cappedTime) or (endIndex == beginIndex):
            break

    return endIndex, time


def sampleOfOptions(verifTimes, indices, cappedTime, explorationPercentage = 1):
    # Returns a list of indices of adaptation options within the bound time and exploration constraint
    if len(indices) == 0:
        return []
    
    totalVerifTime = sum(np.array(verifTimes)[indices])
    if totalVerifTime < cappedTime and explorationPercentage == 1:
        return indices
    
    shuffleList = indices.copy()
    shuffle(shuffleList)

    endInd, _ = determineLearningSamples(np.array(verifTimes)[shuffleList].tolist(), 0, cappedTime)
    if endInd == 0:
        endInd = len(indices)
    percentageOfSamples = (endInd / len(indices))

    if percentageOfSamples <= explorationPercentage:
        return shuffleList[:endInd]
    else:
        # The slice is too big (when looking at the exploration percentage) -> reduce it
        newEnd = int(len(indices) * explorationPercentage)
        return shuffleList[:newEnd]


def getCopyFeatures(begin, end):
    # Returns a deep copy of the features
    return [i.copy() for i in features_global[begin:end]]


if __name__ == '__main__':
    for model, losses, penalties, scalers in classifiers:
        for scaler in scalers:
            for penalty in penalties:
                for loss in losses:
                    for learningTime in [30, 45, 60]:
                        for exploration in [0.05]:
                            simulateModelWithTimes(model, scaler, loss, penalty, amtTrainingCycles=learningTime, explorationPercentage=exploration)


    for model, losses, penalties, scalers in regressors:
        for scaler in scalers:
            for penalty in penalties:
                for loss in losses:
                    for learningTime in [30, 45, 60]:
                        for exploration in [0.05]:
                            simulateModelWithTimes(model, scaler, loss, penalty, amtTrainingCycles=learningTime, explorationPercentage=exploration)



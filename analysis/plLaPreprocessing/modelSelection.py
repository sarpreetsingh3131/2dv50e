import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from util import getDataWithSelectedFeatures
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
import json
from sklearn.ensemble import VotingClassifier
import sys

NB_ADAPTION_PER_CYCLE=216


def main():

    # Too many useless warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    data = getDataWithSelectedFeatures()
    features = data["features"]
    targets = data["targets"]
    rn = 10

    # the robust scaler works better for outliers
    # I use the default parameters, but you could adjust them and leave them in the scaler list
    # and the algorithm will still work.
    scalers = [ ("MaxAbs", sp.MaxAbsScaler()), 
                ("MinMax", sp.MinMaxScaler()), 
                ("Robust", sp.RobustScaler()), 
                ("Standard", sp.StandardScaler()),
                ("None", None)]


    #TODO: should we transform the data with the transformers?

    #Because we online learn, it does not seem fair
    #to let it iterate 5 times over the training data
    # Thats why I will set it to 1
    IT = 1
    to = None

    # It looks like dimensionality reduction is not available for
    # online learning and seem to be quit expensive.
    #dimReduction = [None]
    # TODO: add the dim reduction and classifiers
    classifiers = [ 
                    # Stochastic gradient descent classifiers.
                    #TODO: include alpha, is this C? also for perceptron
                    #TODO: set max_iter for everything to 1, because it would 
                    # be cheating for online learning no?
                    # the default is 5

                    ('SGD-h-l1--4', SGDClassifier(loss='hinge', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l1--4', SGDClassifier(loss='log', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l1--4', SGDClassifier(loss='modified_huber', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l1--4', SGDClassifier(loss='squared_hinge', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l1--4', SGDClassifier(loss='perceptron', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-l2--4', SGDClassifier(loss='hinge', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l2--4', SGDClassifier(loss='log', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l2--4', SGDClassifier(loss='modified_huber', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l2--4', SGDClassifier(loss='squared_hinge', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l2--4', SGDClassifier(loss='perceptron', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-e--4', SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-e--4', SGDClassifier(loss='log', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-e--4', SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-e--4', SGDClassifier(loss='squared_hinge', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-e--4', SGDClassifier(loss='perceptron', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-n--4', SGDClassifier(loss='hinge', penalty='none', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-n--4', SGDClassifier(loss='log', penalty='none', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-n--4', SGDClassifier(loss='modified_huber', penalty='none', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-n--4', SGDClassifier(loss='squared_hinge', penalty='none', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-n--4', SGDClassifier(loss='perceptron', penalty='none', max_iter=IT, tol=to, random_state=rn)),
                    
                    # other regularisation params

                    ('SGD-h-l1--6', SGDClassifier(loss='hinge', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l1--6', SGDClassifier(loss='log', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l1--6', SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l1--6', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l1--6', SGDClassifier(loss='perceptron', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-l2--6', SGDClassifier(loss='hinge', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l2--6', SGDClassifier(loss='log', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l2--6', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l2--6', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l2--6', SGDClassifier(loss='perceptron', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-e--6', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-e--6', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-e--6', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-e--6', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-e--6', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-n--6', SGDClassifier(loss='hinge', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-n--6', SGDClassifier(loss='log', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-n--6', SGDClassifier(loss='modified_huber', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-n--6', SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-n--6', SGDClassifier(loss='perceptron', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-h-l1--2', SGDClassifier(loss='hinge', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l1--2', SGDClassifier(loss='log', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l1--2', SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l1--2', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l1--2', SGDClassifier(loss='perceptron', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-l2--2', SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l2--2', SGDClassifier(loss='log', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l2--2', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l2--2', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l2--2', SGDClassifier(loss='perceptron', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-e--2', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-e--2', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-e--2', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-e--2', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-e--2', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-n--2', SGDClassifier(loss='hinge', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-n--2', SGDClassifier(loss='log', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-n--2', SGDClassifier(loss='modified_huber', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-n--2', SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-n--2', SGDClassifier(loss='perceptron', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    
                    ('SGD-h-l1--1', SGDClassifier(loss='hinge', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l1--1', SGDClassifier(loss='log', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l1--1', SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l1--1', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l1--1', SGDClassifier(loss='perceptron', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-l2--1', SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l2--1', SGDClassifier(loss='log', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l2--1', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l2--1', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l2--1', SGDClassifier(loss='perceptron', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-e--1', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-e--1', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-e--1', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-e--1', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-e--1', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-n--1', SGDClassifier(loss='hinge', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-n--1', SGDClassifier(loss='log', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-n--1', SGDClassifier(loss='modified_huber', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-n--1', SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-n--1', SGDClassifier(loss='perceptron', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-h-l1-0', SGDClassifier(loss='hinge', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l1-0', SGDClassifier(loss='log', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l1-0', SGDClassifier(loss='modified_huber', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l1-0', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l1-0', SGDClassifier(loss='perceptron', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-l2-0', SGDClassifier(loss='hinge', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l2-0', SGDClassifier(loss='log', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l2-0', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l2-0', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l2-0', SGDClassifier(loss='perceptron', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-e-0', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-e-0', SGDClassifier(loss='log', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-e-0', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-e-0', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-e-0', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-n-0', SGDClassifier(loss='hinge', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-n-0', SGDClassifier(loss='log', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-n-0', SGDClassifier(loss='modified_huber', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-n-0', SGDClassifier(loss='squared_hinge', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-n-0', SGDClassifier(loss='perceptron', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),


                    ('SGD-h-l1-1', SGDClassifier(loss='hinge', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l1-1', SGDClassifier(loss='log', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l1-1', SGDClassifier(loss='modified_huber', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l1-1', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l1-1', SGDClassifier(loss='perceptron', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-l2-1', SGDClassifier(loss='hinge', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l2-1', SGDClassifier(loss='log', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l2-1', SGDClassifier(loss='modified_huber', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l2-1', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l2-1', SGDClassifier(loss='perceptron', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-e-1', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-e-1', SGDClassifier(loss='log', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-e-1', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-e-1', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-e-1', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-n-1', SGDClassifier(loss='hinge', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-n-1', SGDClassifier(loss='log', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-n-1', SGDClassifier(loss='modified_huber', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-n-1', SGDClassifier(loss='squared_hinge', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-n-1', SGDClassifier(loss='perceptron', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-h-l1-2', SGDClassifier(loss='hinge', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l1-2', SGDClassifier(loss='log', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l1-2', SGDClassifier(loss='modified_huber', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l1-2', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l1-2', SGDClassifier(loss='perceptron', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-l2-2', SGDClassifier(loss='hinge', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-l2-2', SGDClassifier(loss='log', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-l2-2', SGDClassifier(loss='modified_huber', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-l2-2', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-l2-2', SGDClassifier(loss='perceptron', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-e-2', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-e-2', SGDClassifier(loss='log', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-e-2', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-e-2', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-e-2', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-h-n-2', SGDClassifier(loss='hinge', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-l-n-2', SGDClassifier(loss='log', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-mh-n-2', SGDClassifier(loss='modified_huber', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sh-n-2', SGDClassifier(loss='squared_hinge', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-p-n-2', SGDClassifier(loss='perceptron', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),

                    # Below classifier use regression loss functions,
                    # but the sklearn docs say it can be usefull for regression too
                    ('SGD-sl-l1--4', SGDClassifier(loss='squared_loss', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l1--4', SGDClassifier(loss='huber', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l1--4', SGDClassifier(loss='epsilon_insensitive', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l1--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-l2--4', SGDClassifier(loss='squared_loss', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l2--4', SGDClassifier(loss='huber', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l2--4', SGDClassifier(loss='epsilon_insensitive', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l2--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-e--4', SGDClassifier(loss='squared_loss', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-e--4', SGDClassifier(loss='huber', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-e--4', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-e--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-n--4', SGDClassifier(loss='squared_loss', penalty='none', max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-n--4', SGDClassifier(loss='huber', penalty='none', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-n--4', SGDClassifier(loss='epsilon_insensitive', penalty='none', max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-n--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', max_iter=IT, tol=to, random_state=rn)),
                    
                    ('SGD-sl-l1--6', SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l1--6', SGDClassifier(loss='huber', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l1--6', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l1--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-l2--6', SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l2--6', SGDClassifier(loss='huber', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l2--6', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l2--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-e--6', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-e--6', SGDClassifier(loss='huber', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-e--6', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-e--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-n--6', SGDClassifier(loss='squared_loss', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-n--6', SGDClassifier(loss='huber', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-n--6', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-n--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-sl-l1--2', SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l1--2', SGDClassifier(loss='huber', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l1--2', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l1--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-l2--2', SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l2--2', SGDClassifier(loss='huber', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l2--2', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l2--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-e--2', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-e--2', SGDClassifier(loss='huber', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-e--2', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-e--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-n--2', SGDClassifier(loss='squared_loss', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-n--2', SGDClassifier(loss='huber', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-n--2', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-n--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-sl-l1--1', SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l1--1', SGDClassifier(loss='huber', penalty='l1', alpha=0.1, tol=to, random_state=rn)),
                    ('SGD-ei-l1--1', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l1--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-l2--1', SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l2--1', SGDClassifier(loss='huber', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l2--1', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l2--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-e--1', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-e--1', SGDClassifier(loss='huber', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-e--1', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-e--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-n--1', SGDClassifier(loss='squared_loss', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-n--1', SGDClassifier(loss='huber', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-n--1', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-n--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-sl-l1-0', SGDClassifier(loss='squared_loss', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l1-0', SGDClassifier(loss='huber', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l1-0', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l1-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-l2-0', SGDClassifier(loss='squared_loss', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l2-0', SGDClassifier(loss='huber', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l2-0', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l2-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-e-0', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-e-0', SGDClassifier(loss='huber', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-e-0', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-e-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-n-0', SGDClassifier(loss='squared_loss', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-n-0', SGDClassifier(loss='huber', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-n-0', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-n-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-sl-l1-1', SGDClassifier(loss='squared_loss', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l1-1', SGDClassifier(loss='huber', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l1-1', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l1-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-l2-1', SGDClassifier(loss='squared_loss', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l2-1', SGDClassifier(loss='huber', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l2-1', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l2-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-e-1', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-e-1', SGDClassifier(loss='huber', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-e-1', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-e-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-n-1', SGDClassifier(loss='squared_loss', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-n-1', SGDClassifier(loss='huber', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-n-1', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-n-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),

                    ('SGD-sl-l1-2', SGDClassifier(loss='squared_loss', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l1-2', SGDClassifier(loss='huber', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l1-2', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l1-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-l2-2', SGDClassifier(loss='squared_loss', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-l2-2', SGDClassifier(loss='huber', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-l2-2', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-l2-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-e-2', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-e-2', SGDClassifier(loss='huber', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-e-2', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-e-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sl-n-2', SGDClassifier(loss='squared_loss', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
        	        ('SGD-hu-n-2', SGDClassifier(loss='huber', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-ei-n-2', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('SGD-sei-n-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),

                    #  Perceptron 
                    ('Perceptron-l1--4', Perceptron(penalty='l1', max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-l2--4', Perceptron(penalty='l2', max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-e--4', Perceptron(penalty='elasticnet', max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-n--4', Perceptron(penalty='none', max_iter=IT, tol=to, random_state=rn)),

                    ('Perceptron-l1--6', Perceptron(penalty='l1', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-l2--6', Perceptron(penalty='l2', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-e--6', Perceptron(penalty='elasticnet', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-n--6', Perceptron(penalty='none', alpha=0.000001, max_iter=IT, tol=to, random_state=rn)),

                    ('Perceptron-l1--2', Perceptron(penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-l2--2', Perceptron(penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-e--2', Perceptron(penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-n--2', Perceptron(penalty='none', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),

                    ('Perceptron-l1--1', Perceptron(penalty='l1', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-l2--1', Perceptron(penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-e--1', Perceptron(penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-n--1', Perceptron(penalty='none', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),

                    ('Perceptron-l1--0', Perceptron(penalty='l1', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-l2--0', Perceptron(penalty='l2', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-e--0', Perceptron(penalty='elasticnet', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-n--0', Perceptron(penalty='none', alpha=1.0, max_iter=IT, tol=to, random_state=rn)),

                    ('Perceptron-l1-1', Perceptron(penalty='l1', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-l2-1', Perceptron(penalty='l2', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-e-1', Perceptron(penalty='elasticnet', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-n-1', Perceptron(penalty='none', alpha=10.0, max_iter=IT, tol=to, random_state=rn)),

                    ('Perceptron-l1-2', Perceptron(penalty='l1', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-l2-2', Perceptron(penalty='l2', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-e-2', Perceptron(penalty='elasticnet', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),
                    ('Perceptron-n-2', Perceptron(penalty='none', alpha=100.0, max_iter=IT, tol=to, random_state=rn)),



                    # Passive aggressive 
                    # TODO: what is step size and maybe vary
                    ('PA-I-0', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, tol=to, random_state=rn)),
                    ('PA-II-0', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, tol=to, random_state=rn)),

                    ('PA-I--1', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.1, tol=to, random_state=rn)),
                    ('PA-II--1', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.1, tol=to, random_state=rn)),

                    ('PA-I--2', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.01, tol=to, random_state=rn)),
                    ('PA-II--2', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.01, tol=to, random_state=rn)),

                    ('PA-I--3', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.001, tol=to, random_state=rn)),
                    ('PA-II--3', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.001, tol=to, random_state=rn)),

                    ('PA-I--4', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.0001, tol=to, random_state=rn)),
                    ('PA-II--4', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.0001, tol=to, random_state=rn)),

                    ('PA-I-1', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=10.0, tol=to, random_state=rn)),
                    ('PA-II-1', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=10.0, tol=to, random_state=rn)),

                    ('PA-I-2', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=100.0, tol=to, random_state=rn)),
                    ('PA-II-2', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=100.0, tol=to, random_state=rn)),


                    # Varying step size
                    # I do not know what it is, but take a look
                    # at this website
                    # http://koaning.io/passive-agressive-algorithms.html

                    
                    #The following are multiclass able classifiers who can online learn 
                    #are listed below:
                    #They were grabbed from here
                    #https://scikit-learn.org/stable/modules/multiclass.html

                    #Other possible online multiclass classifiers:
                    #sklearn.naive_bayes.BernoulliNB
                    #sklearn.naive_bayes.GaussianNB

                    #TODO: alpha varieren die de smoothing parameter voorsteld
                    ("BNB-1", BernoulliNB(alpha=1.0)),
                    ("BNB-0.75", BernoulliNB(alpha=0.75)),
                    ("BNB-0.5", BernoulliNB(alpha=0.5)),
                    ("BNB-0.25", BernoulliNB(alpha=0.25)),
                    ("BNB-0.0001", BernoulliNB(alpha=0.0001)),

                    #THe GaussianNB did not really have parameters to vary
                    ("GNB", GaussianNB())
                    
                    ]




    # Make data ready
    kf = 10

    skf = StratifiedKFold(n_splits=kf, shuffle=True, random_state=10)
    
    #(trainIndeces, testIndices) = skf.split(features,targets)
    ks = skf.split(features,targets)

    trainSetsF = []
    trainSetsT = []
    testSetsF = []
    testSetsT = []

    tsf = []
    tst = []
    tesf = []
    tesk = []

    for (tr,te) in ks:
        tsf = []
        tst = []
        tesf = []
        tesk = []
        for i in tr:
            tsf.append(features[i])
            tst.append(targets[i])
        for i in te:
            tesf.append(features[i])
            tesk.append(targets[i])
        trainSetsF.append(tsf.copy())
        trainSetsT.append(tst.copy())
        testSetsF.append(tesf.copy())
        testSetsT.append(tesk.copy())
    
    #print(len(features))
    #print(len(features[0]))
    #print(len(targets))
    #print(len(trainSetsF))
    #print(len(testSetsF))
    #print(len(trainSetsF[1]))
    #print(len(testSetsF[1]))

    print("Starting loop.")
    size = len(scalers)*len(classifiers)
    print("0/"+str(size)+" completed",end='\r')
    

    cNames = []
    sNames = []
    times = []
    scores = []
    means = []
    stds = []
    scAndCl=[]

    z = 0
    faultPerClass = []
    
    for scaler in scalers:

        #for dimReducer in dimReduction:
        trainSetsFs = []
        testSetsFs = []
        if (not(scaler[1] is None)):
            for i in range(len(trainSetsF)):
                trainSetsFs.append( scaler[1].fit_transform(trainSetsF[i].copy()))
            for i in range(len(testSetsF)):
                testSetsFs.append(scaler[1].transform(testSetsF[i].copy()))
        else:
            trainSetsFs = trainSetsF.copy()
            testSetsFs = testSetsF.copy()


        for classifier in classifiers:

            start = time.time()

            a = []
            faults = [0,0,0,0]

            for trainF, trainT, testF, testT in zip(trainSetsFs, trainSetsT, testSetsFs, testSetsT):
                
                m = 0

                fr = [0,0,0,0]                

                #if (not(scaler[1] is None)):
                    #trainF = scaler[1].fit_transform(trainF)
                    #testF = scaler[1].transform(testF)

                # try to partial fit everything

                # TODO: change to fit, because to slow, maybe it helps
                #for caseF, caseT in zip(trainF, trainT):
                #    caseT = np.array([caseT])
                #    classifier[1].partial_fit(X=caseF.reshape(1,-1), y=caseT, classes=np.array([0,1,2,3]))

                classifier[1].fit(trainF, trainT)

                # I wont use score here, because it returns a harsh
                # metric which will always be low.
                # I will do it manually

                pred = classifier[1].predict(testF)

                for p,t in zip(pred, testT):
                    if (p == t):
                        m += 1
                    else:
                        fr[t] += 1

                valasdf, counts = np.unique(np.array(testT), return_counts=True)
                #print(fr)
                for i in range(len(faults)):
                    faults[i] += fr[i] / counts[i]
                m = m / len(testT)
                a.append(m)
                #print(faults)

            me = np.mean(np.array(a))
            va = np.std(np.array(a))

            sec = time.time()-start

            cNames.append(classifier[0])
            sNames.append(scaler[0])
            times.append(sec)
            scores.append(a.copy())
            means.append(me)
            stds.append(va)
            scAndCl.append((scaler[1],classifier[1]))
            
            
            #print(faults)
            #print(counts)
            #for wrong, total in zip(faults, counts):
                #wrong = wrong / total
            for i in range(len(faults)):
                faults[i] = faults[i] / kf

            #print(faults)

            faultPerClass.append(faults.copy())

            z += 1

            ms = np.mean(np.array(times))

            print(str(z)+"/"+str(size)+" completed. This: "+str(sec)+" sec. Average: "+str(ms)+" sec.       ", end='\r')

    print("\nCompleted analysis. \n Constructing data about top 6.")

    top6 = [-1,-1,-1,-1,-1,-1]

    for i in range(0, len(means)):
        j = 0
        while( j < len(top6) and ( top6[j] == -1 or means[i] >= means[top6[j]]) ):
            j += 1
        if (j != 0):
            k = j - 1
            h = 1
            while( k >= h):
                top6[h-1] = top6[h]
                h += 1
            top6[k] = i

    step = 1
    end = 100
    if (end*NB_ADAPTION_PER_CYCLE > len(targets)):
        end = (((len(targets) // NB_ADAPTION_PER_CYCLE ) -1 ) // step) * step

    gaps = range(step,end+1,step)
    vals = []
    for i in range(0,len(top6)):
        scal, clas = scAndCl[top6[i]]
        
        # Probleem  
        val = []
        
        for j in gaps:
            sepe = (NB_ADAPTION_PER_CYCLE * j) - 1
            seps = sepe + 1
            fs = features[0:sepe]
            ts = features[seps:]
            if (not(scal is None)):
                fs = scal.fit_transform(fs)
                ts = scal.transform(ts)
            
            clas.fit(fs, targets[0:sepe])
            pred = clas.predict(ts)

            q = 0
            for p,t in zip(pred, targets[seps:]):
                if (p == t):
                    q +=1
            val.append(1-(q/len(pred)))
        vals.append(np.array(val))

    fig = plt.figure(figsize=(12,10))
    
    indx = np.argsort(np.array(means))

    plt.subplot(4,1,1)

    for i in range(len(indx)-1,len(indx)-102,-1):
        co = 'w'
        if (sNames[indx[i]] == scalers[0][0]):
            co = 'r'
        elif (sNames[indx[i]] == scalers[1][0]):
            co = 'b'
        elif (sNames[indx[i]] == scalers[2][0]):
            co = 'g'
        elif (sNames[indx[i]] == scalers[3][0]):
            co = 'm'
        elif (sNames[indx[i]] == scalers[4][0]):
            co = 'black'
        plt.bar(i + 1, means[indx[i]], color=co, label=sNames[indx[i]])

    plt.ylabel("Prop. right")
    plt.xlabel("top 100 scalers x classifiers")
    plt.title("Top 100 performance of all scalers x classifiers")


    for i in range(len(top6) -1 ,-1,-1):
        #plt.subplot(6,3,3*(5-i)+1)
        plt.subplot(4,6,6-i+6)
        plt.bar(list(range(1,kf+1)), scores[top6[i]], label="Scores over all folds")
        plt.hlines(means[top6[i]], 0, kf+1, 'r', label="Mean")
        plt.ylabel("1-error rate")
        plt.xlabel("fold")
        plt.title(str(6-i)+") "+str(cNames[top6[i]])+"\n"+str(sNames[top6[i]])+"\n Mean: "+str(round(means[top6[i]], 6))+"\n std: "+str(round(stds[top6[i]],8))+"\n Time: "+str(round(times[top6[i]],5))+" secs")

        #plt.subplot(6,3, 3*(5-i)+2)
        plt.subplot(4,6, 6-i + 12)
        plt.plot(gaps, vals[i])
        plt.ylabel("error rate")
        plt.xlabel("Nb of adaptation cycles")
        plt.title("Error conversion")

        #plt.subplot(6,3, 3*(5-i)+3)
        plt.subplot(4,6, 6-i + 18)
        plt.bar( [0,1,2,3], faultPerClass[top6[i]], label="0: No goal \n 1: Packet loss \n 2: Latency \n 3: Both")
        plt.xlabel("Class")
        plt.ylabel("% wrong")
        plt.title("Faults")


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO: ensemble learning sklearn.ensemble.VotingClassifier
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #ens = VotingClassifier([("clas 1", classifiers[1]),("clas2", classifiers[2])])
    #ens.predict(features)

    plt.tight_layout()
    plt.show()

    x = "asdf"
    while( x != 'y' and x != 'n' ):
        x = input("Do you want to save and overwrite previous data? [y/n]\n")
    if (x == 'y'):
        

        fig.savefig('modelSelection.eps', format='eps', dpi=1200)
        fig.savefig('modelSelection.svg', format='svg', dpi=1200)
        fig.savefig('modelSelection.png', format='png', dpi=1200)


        strin = ""
        
        #TODO: put classifier in ensemble that al have 1 weakness for a different class, so they complement each other (1 says wrong, 3 say right for example)
        for i in indx:
            strin =  str(i+1) +") Scaler: "+ sNames[i] + ", classifier: "+ cNames[i] + ", mean: "+ str(means[i]) + ", class wrong: " + str(faultPerClass[i]) + "\n" + strin 
        

        with open("msRank.txt", "w") as f:
            f.write(strin)

                
        data = {}
        data["cName"] = cNames
        data["sName"] = sNames
        data["time"] = times
        data["score"] = scores
        data["mean"] = means
        data["std"] = stds
        data["faultPerClass"] = faultPerClass


        with open("modelSelectionResults.json","w") as f:
            json.dump(data, f)



                

                


    #if scaler is None:


main()
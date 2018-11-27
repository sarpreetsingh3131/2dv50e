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

NB_ADAPTION_PER_CYCLE=216

def main():

    data = getDataWithSelectedFeatures()
    features = data["features"]
    targets = data["targets"]

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

                    ('SGD-h-l1--4', SGDClassifier(loss='hinge', penalty='l1', max_iter=IT)),
        	        ('SGD-l-l1--4', SGDClassifier(loss='log', penalty='l1', max_iter=IT)),
                    ('SGD-mh-l1--4', SGDClassifier(loss='modified_huber', penalty='l1', max_iter=IT)),
                    ('SGD-sh-l1--4', SGDClassifier(loss='squared_hinge', penalty='l1', max_iter=IT)),
                    ('SGD-p-l1--4', SGDClassifier(loss='perceptron', penalty='l1', max_iter=IT)),
                    ('SGD-h-l2--4', SGDClassifier(loss='hinge', penalty='l2', max_iter=IT)),
        	        ('SGD-l-l2--4', SGDClassifier(loss='log', penalty='l2', max_iter=IT)),
                    ('SGD-mh-l2--4', SGDClassifier(loss='modified_huber', penalty='l2', max_iter=IT)),
                    ('SGD-sh-l2--4', SGDClassifier(loss='squared_hinge', penalty='l2', max_iter=IT)),
                    ('SGD-p-l2--4', SGDClassifier(loss='perceptron', penalty='l2', max_iter=IT)),
                    ('SGD-h-e--4', SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=IT)),
        	        ('SGD-l-e--4', SGDClassifier(loss='log', penalty='elasticnet', max_iter=IT)),
                    ('SGD-mh-e--4', SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=IT)),
                    ('SGD-sh-e--4', SGDClassifier(loss='squared_hinge', penalty='elasticnet', max_iter=IT)),
                    ('SGD-p-e--4', SGDClassifier(loss='perceptron', penalty='elasticnet', max_iter=IT)),
                    ('SGD-h-n--4', SGDClassifier(loss='hinge', penalty='none', max_iter=IT)),
        	        ('SGD-l-n--4', SGDClassifier(loss='log', penalty='none', max_iter=IT)),
                    ('SGD-mh-n--4', SGDClassifier(loss='modified_huber', penalty='none', max_iter=IT)),
                    ('SGD-sh-n--4', SGDClassifier(loss='squared_hinge', penalty='none', max_iter=IT)),
                    ('SGD-p-n--4', SGDClassifier(loss='perceptron', penalty='none', max_iter=IT)),
                    
                    # other regularisation params

                    ('SGD-h-l1--6', SGDClassifier(loss='hinge', penalty='l1', alpha=0.000001, max_iter=IT)),
        	        ('SGD-l-l1--6', SGDClassifier(loss='log', penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('SGD-mh-l1--6', SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('SGD-sh-l1--6', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('SGD-p-l1--6', SGDClassifier(loss='perceptron', penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('SGD-h-l2--6', SGDClassifier(loss='hinge', penalty='l2', alpha=0.000001, max_iter=IT)),
        	        ('SGD-l-l2--6', SGDClassifier(loss='log', penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('SGD-mh-l2--6', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('SGD-sh-l2--6', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('SGD-p-l2--6', SGDClassifier(loss='perceptron', penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('SGD-h-e--6', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
        	        ('SGD-l-e--6', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('SGD-mh-e--6', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('SGD-sh-e--6', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('SGD-p-e--6', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('SGD-h-n--6', SGDClassifier(loss='hinge', penalty='none', alpha=0.000001, max_iter=IT)),
        	        ('SGD-l-n--6', SGDClassifier(loss='log', penalty='none', alpha=0.000001, max_iter=IT)),
                    ('SGD-mh-n--6', SGDClassifier(loss='modified_huber', penalty='none', alpha=0.000001, max_iter=IT)),
                    ('SGD-sh-n--6', SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.000001, max_iter=IT)),
                    ('SGD-p-n--6', SGDClassifier(loss='perceptron', penalty='none', alpha=0.000001, max_iter=IT)),

                    ('SGD-h-l1--2', SGDClassifier(loss='hinge', penalty='l1', alpha=0.01, max_iter=IT)),
        	        ('SGD-l-l1--2', SGDClassifier(loss='log', penalty='l1', alpha=0.01, max_iter=IT)),
                    ('SGD-mh-l1--2', SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.01, max_iter=IT)),
                    ('SGD-sh-l1--2', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=0.01, max_iter=IT)),
                    ('SGD-p-l1--2', SGDClassifier(loss='perceptron', penalty='l1', alpha=0.01, max_iter=IT)),
                    ('SGD-h-l2--2', SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, max_iter=IT)),
        	        ('SGD-l-l2--2', SGDClassifier(loss='log', penalty='l2', alpha=0.01, max_iter=IT)),
                    ('SGD-mh-l2--2', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.01, max_iter=IT)),
                    ('SGD-sh-l2--2', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.01, max_iter=IT)),
                    ('SGD-p-l2--2', SGDClassifier(loss='perceptron', penalty='l2', alpha=0.01, max_iter=IT)),
                    ('SGD-h-e--2', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.01, max_iter=IT)),
        	        ('SGD-l-e--2', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('SGD-mh-e--2', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('SGD-sh-e--2', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('SGD-p-e--2', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('SGD-h-n--2', SGDClassifier(loss='hinge', penalty='none', alpha=0.01, max_iter=IT)),
        	        ('SGD-l-n--2', SGDClassifier(loss='log', penalty='none', alpha=0.01, max_iter=IT)),
                    ('SGD-mh-n--2', SGDClassifier(loss='modified_huber', penalty='none', alpha=0.01, max_iter=IT)),
                    ('SGD-sh-n--2', SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.01, max_iter=IT)),
                    ('SGD-p-n--2', SGDClassifier(loss='perceptron', penalty='none', alpha=0.01, max_iter=IT)),
                    
                    ('SGD-h-l1--1', SGDClassifier(loss='hinge', penalty='l1', alpha=0.1, max_iter=IT)),
        	        ('SGD-l-l1--1', SGDClassifier(loss='log', penalty='l1', alpha=0.1, max_iter=IT)),
                    ('SGD-mh-l1--1', SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.1, max_iter=IT)),
                    ('SGD-sh-l1--1', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=0.1, max_iter=IT)),
                    ('SGD-p-l1--1', SGDClassifier(loss='perceptron', penalty='l1', alpha=0.1, max_iter=IT)),
                    ('SGD-h-l2--1', SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, max_iter=IT)),
        	        ('SGD-l-l2--1', SGDClassifier(loss='log', penalty='l2', alpha=0.1, max_iter=IT)),
                    ('SGD-mh-l2--1', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.1, max_iter=IT)),
                    ('SGD-sh-l2--1', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.1, max_iter=IT)),
                    ('SGD-p-l2--1', SGDClassifier(loss='perceptron', penalty='l2', alpha=0.1, max_iter=IT)),
                    ('SGD-h-e--1', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.1, max_iter=IT)),
        	        ('SGD-l-e--1', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('SGD-mh-e--1', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('SGD-sh-e--1', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('SGD-p-e--1', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('SGD-h-n--1', SGDClassifier(loss='hinge', penalty='none', alpha=0.1, max_iter=IT)),
        	        ('SGD-l-n--1', SGDClassifier(loss='log', penalty='none', alpha=0.1, max_iter=IT)),
                    ('SGD-mh-n--1', SGDClassifier(loss='modified_huber', penalty='none', alpha=0.1, max_iter=IT)),
                    ('SGD-sh-n--1', SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.1, max_iter=IT)),
                    ('SGD-p-n--1', SGDClassifier(loss='perceptron', penalty='none', alpha=0.1, max_iter=IT)),

                    ('SGD-h-l1-0', SGDClassifier(loss='hinge', penalty='l1', alpha=1.0, max_iter=IT)),
        	        ('SGD-l-l1-0', SGDClassifier(loss='log', penalty='l1', alpha=1.0, max_iter=IT)),
                    ('SGD-mh-l1-0', SGDClassifier(loss='modified_huber', penalty='l1', alpha=1.0, max_iter=IT)),
                    ('SGD-sh-l1-0', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=1.0, max_iter=IT)),
                    ('SGD-p-l1-0', SGDClassifier(loss='perceptron', penalty='l1', alpha=1.0, max_iter=IT)),
                    ('SGD-h-l2-0', SGDClassifier(loss='hinge', penalty='l2', alpha=1.0, max_iter=IT)),
        	        ('SGD-l-l2-0', SGDClassifier(loss='log', penalty='l2', alpha=1.0, max_iter=IT)),
                    ('SGD-mh-l2-0', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1.0, max_iter=IT)),
                    ('SGD-sh-l2-0', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1.0, max_iter=IT)),
                    ('SGD-p-l2-0', SGDClassifier(loss='perceptron', penalty='l2', alpha=1.0, max_iter=IT)),
                    ('SGD-h-e-0', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=1.0, max_iter=IT)),
        	        ('SGD-l-e-0', SGDClassifier(loss='log', penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('SGD-mh-e-0', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('SGD-sh-e-0', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('SGD-p-e-0', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('SGD-h-n-0', SGDClassifier(loss='hinge', penalty='none', alpha=1.0, max_iter=IT)),
        	        ('SGD-l-n-0', SGDClassifier(loss='log', penalty='none', alpha=1.0, max_iter=IT)),
                    ('SGD-mh-n-0', SGDClassifier(loss='modified_huber', penalty='none', alpha=1.0, max_iter=IT)),
                    ('SGD-sh-n-0', SGDClassifier(loss='squared_hinge', penalty='none', alpha=1.0, max_iter=IT)),
                    ('SGD-p-n-0', SGDClassifier(loss='perceptron', penalty='none', alpha=1.0, max_iter=IT)),


                    ('SGD-h-l1-1', SGDClassifier(loss='hinge', penalty='l1', alpha=10.0, max_iter=IT)),
        	        ('SGD-l-l1-1', SGDClassifier(loss='log', penalty='l1', alpha=10.0, max_iter=IT)),
                    ('SGD-mh-l1-1', SGDClassifier(loss='modified_huber', penalty='l1', alpha=10.0, max_iter=IT)),
                    ('SGD-sh-l1-1', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=10.0, max_iter=IT)),
                    ('SGD-p-l1-1', SGDClassifier(loss='perceptron', penalty='l1', alpha=10.0, max_iter=IT)),
                    ('SGD-h-l2-1', SGDClassifier(loss='hinge', penalty='l2', alpha=10.0, max_iter=IT)),
        	        ('SGD-l-l2-1', SGDClassifier(loss='log', penalty='l2', alpha=10.0, max_iter=IT)),
                    ('SGD-mh-l2-1', SGDClassifier(loss='modified_huber', penalty='l2', alpha=10.0, max_iter=IT)),
                    ('SGD-sh-l2-1', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=10.0, max_iter=IT)),
                    ('SGD-p-l2-1', SGDClassifier(loss='perceptron', penalty='l2', alpha=10.0, max_iter=IT)),
                    ('SGD-h-e-1', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=10.0, max_iter=IT)),
        	        ('SGD-l-e-1', SGDClassifier(loss='log', penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('SGD-mh-e-1', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('SGD-sh-e-1', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('SGD-p-e-1', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('SGD-h-n-1', SGDClassifier(loss='hinge', penalty='none', alpha=10.0, max_iter=IT)),
        	        ('SGD-l-n-1', SGDClassifier(loss='log', penalty='none', alpha=10.0, max_iter=IT)),
                    ('SGD-mh-n-1', SGDClassifier(loss='modified_huber', penalty='none', alpha=10.0, max_iter=IT)),
                    ('SGD-sh-n-1', SGDClassifier(loss='squared_hinge', penalty='none', alpha=10.0, max_iter=IT)),
                    ('SGD-p-n-1', SGDClassifier(loss='perceptron', penalty='none', alpha=10.0, max_iter=IT)),

                    ('SGD-h-l1-2', SGDClassifier(loss='hinge', penalty='l1', alpha=100.0, max_iter=IT)),
        	        ('SGD-l-l1-2', SGDClassifier(loss='log', penalty='l1', alpha=100.0, max_iter=IT)),
                    ('SGD-mh-l1-2', SGDClassifier(loss='modified_huber', penalty='l1', alpha=100.0, max_iter=IT)),
                    ('SGD-sh-l1-2', SGDClassifier(loss='squared_hinge', penalty='l1', alpha=100.0, max_iter=IT)),
                    ('SGD-p-l1-2', SGDClassifier(loss='perceptron', penalty='l1', alpha=100.0, max_iter=IT)),
                    ('SGD-h-l2-2', SGDClassifier(loss='hinge', penalty='l2', alpha=100.0, max_iter=IT)),
        	        ('SGD-l-l2-2', SGDClassifier(loss='log', penalty='l2', alpha=100.0, max_iter=IT)),
                    ('SGD-mh-l2-2', SGDClassifier(loss='modified_huber', penalty='l2', alpha=100.0, max_iter=IT)),
                    ('SGD-sh-l2-2', SGDClassifier(loss='squared_hinge', penalty='l2', alpha=100.0, max_iter=IT)),
                    ('SGD-p-l2-2', SGDClassifier(loss='perceptron', penalty='l2', alpha=100.0, max_iter=IT)),
                    ('SGD-h-e-2', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=100.0, max_iter=IT)),
        	        ('SGD-l-e-2', SGDClassifier(loss='log', penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('SGD-mh-e-2', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('SGD-sh-e-2', SGDClassifier(loss='squared_hinge', penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('SGD-p-e-2', SGDClassifier(loss='perceptron', penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('SGD-h-n-2', SGDClassifier(loss='hinge', penalty='none', alpha=100.0, max_iter=IT)),
        	        ('SGD-l-n-2', SGDClassifier(loss='log', penalty='none', alpha=100.0, max_iter=IT)),
                    ('SGD-mh-n-2', SGDClassifier(loss='modified_huber', penalty='none', alpha=100.0, max_iter=IT)),
                    ('SGD-sh-n-2', SGDClassifier(loss='squared_hinge', penalty='none', alpha=100.0, max_iter=IT)),
                    ('SGD-p-n-2', SGDClassifier(loss='perceptron', penalty='none', alpha=100.0, max_iter=IT)),

                    # Below classifier use regression loss functions,
                    # but the sklearn docs say it can be usefull for regression too
                    ('SGD-sl-l1--4', SGDClassifier(loss='squared_loss', penalty='l1', max_iter=IT)),
        	        ('SGD-hu-l1--4', SGDClassifier(loss='huber', penalty='l1', max_iter=IT)),
                    ('SGD-ei-l1--4', SGDClassifier(loss='epsilon_insensitive', penalty='l1', max_iter=IT)),
                    ('SGD-sei-l1--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', max_iter=IT)),
                    ('SGD-sl-l2--4', SGDClassifier(loss='squared_loss', penalty='l2', max_iter=IT)),
        	        ('SGD-hu-l2--4', SGDClassifier(loss='huber', penalty='l2', max_iter=IT)),
                    ('SGD-ei-l2--4', SGDClassifier(loss='epsilon_insensitive', penalty='l2', max_iter=IT)),
                    ('SGD-sei-l2--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', max_iter=IT)),
                    ('SGD-sl-e--4', SGDClassifier(loss='squared_loss', penalty='elasticnet', max_iter=IT)),
        	        ('SGD-hu-e--4', SGDClassifier(loss='huber', penalty='elasticnet', max_iter=IT)),
                    ('SGD-ei-e--4', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', max_iter=IT)),
                    ('SGD-sei-e--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', max_iter=IT)),
                    ('SGD-sl-n--4', SGDClassifier(loss='squared_loss', penalty='none', max_iter=IT)),
        	        ('SGD-hu-n--4', SGDClassifier(loss='huber', penalty='none', max_iter=IT)),
                    ('SGD-ei-n--4', SGDClassifier(loss='epsilon_insensitive', penalty='none', max_iter=IT)),
                    ('SGD-sei-n--4', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', max_iter=IT)),
                    
                    ('SGD-sl-l1--6', SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.000001, max_iter=IT)),
        	        ('SGD-hu-l1--6', SGDClassifier(loss='huber', penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('SGD-ei-l1--6', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('SGD-sei-l1--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('SGD-sl-l2--6', SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.000001, max_iter=IT)),
        	        ('SGD-hu-l2--6', SGDClassifier(loss='huber', penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('SGD-ei-l2--6', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('SGD-sei-l2--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('SGD-sl-e--6', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
        	        ('SGD-hu-e--6', SGDClassifier(loss='huber', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('SGD-ei-e--6', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('SGD-sei-e--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('SGD-sl-n--6', SGDClassifier(loss='squared_loss', penalty='none', alpha=0.000001, max_iter=IT)),
        	        ('SGD-hu-n--6', SGDClassifier(loss='huber', penalty='none', alpha=0.000001, max_iter=IT)),
                    ('SGD-ei-n--6', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=0.000001, max_iter=IT)),
                    ('SGD-sei-n--6', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=0.000001, max_iter=IT)),

                    ('SGD-sl-l1--2', SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.01, max_iter=IT)),
        	        ('SGD-hu-l1--2', SGDClassifier(loss='huber', penalty='l1', alpha=0.01, max_iter=IT)),
                    ('SGD-ei-l1--2', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=0.01, max_iter=IT)),
                    ('SGD-sei-l1--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=0.01, max_iter=IT)),
                    ('SGD-sl-l2--2', SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.01, max_iter=IT)),
        	        ('SGD-hu-l2--2', SGDClassifier(loss='huber', penalty='l2', alpha=0.01, max_iter=IT)),
                    ('SGD-ei-l2--2', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=0.01, max_iter=IT)),
                    ('SGD-sei-l2--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.01, max_iter=IT)),
                    ('SGD-sl-e--2', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=0.01, max_iter=IT)),
        	        ('SGD-hu-e--2', SGDClassifier(loss='huber', penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('SGD-ei-e--2', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('SGD-sei-e--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('SGD-sl-n--2', SGDClassifier(loss='squared_loss', penalty='none', alpha=0.01, max_iter=IT)),
        	        ('SGD-hu-n--2', SGDClassifier(loss='huber', penalty='none', alpha=0.01, max_iter=IT)),
                    ('SGD-ei-n--2', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=0.01, max_iter=IT)),
                    ('SGD-sei-n--2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=0.01, max_iter=IT)),

                    ('SGD-sl-l1--1', SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.1, max_iter=IT)),
        	        ('SGD-hu-l1--1', SGDClassifier(loss='huber', penalty='l1', alpha=0.1)),
                    ('SGD-ei-l1--1', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=0.1, max_iter=IT)),
                    ('SGD-sei-l1--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=0.1, max_iter=IT)),
                    ('SGD-sl-l2--1', SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.1, max_iter=IT)),
        	        ('SGD-hu-l2--1', SGDClassifier(loss='huber', penalty='l2', alpha=0.1, max_iter=IT)),
                    ('SGD-ei-l2--1', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=0.1, max_iter=IT)),
                    ('SGD-sei-l2--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.1, max_iter=IT)),
                    ('SGD-sl-e--1', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=0.1, max_iter=IT)),
        	        ('SGD-hu-e--1', SGDClassifier(loss='huber', penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('SGD-ei-e--1', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('SGD-sei-e--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('SGD-sl-n--1', SGDClassifier(loss='squared_loss', penalty='none', alpha=0.1, max_iter=IT)),
        	        ('SGD-hu-n--1', SGDClassifier(loss='huber', penalty='none', alpha=0.1, max_iter=IT)),
                    ('SGD-ei-n--1', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=0.1, max_iter=IT)),
                    ('SGD-sei-n--1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=0.1, max_iter=IT)),

                    ('SGD-sl-l1-0', SGDClassifier(loss='squared_loss', penalty='l1', alpha=1.0, max_iter=IT)),
        	        ('SGD-hu-l1-0', SGDClassifier(loss='huber', penalty='l1', alpha=1.0, max_iter=IT)),
                    ('SGD-ei-l1-0', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=1.0, max_iter=IT)),
                    ('SGD-sei-l1-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=1.0, max_iter=IT)),
                    ('SGD-sl-l2-0', SGDClassifier(loss='squared_loss', penalty='l2', alpha=1.0, max_iter=IT)),
        	        ('SGD-hu-l2-0', SGDClassifier(loss='huber', penalty='l2', alpha=1.0, max_iter=IT)),
                    ('SGD-ei-l2-0', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=1.0, max_iter=IT)),
                    ('SGD-sei-l2-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=1.0, max_iter=IT)),
                    ('SGD-sl-e-0', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=1.0, max_iter=IT)),
        	        ('SGD-hu-e-0', SGDClassifier(loss='huber', penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('SGD-ei-e-0', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('SGD-sei-e-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('SGD-sl-n-0', SGDClassifier(loss='squared_loss', penalty='none', alpha=1.0, max_iter=IT)),
        	        ('SGD-hu-n-0', SGDClassifier(loss='huber', penalty='none', alpha=1.0, max_iter=IT)),
                    ('SGD-ei-n-0', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=1.0, max_iter=IT)),
                    ('SGD-sei-n-0', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=1.0, max_iter=IT)),

                    ('SGD-sl-l1-1', SGDClassifier(loss='squared_loss', penalty='l1', alpha=10.0, max_iter=IT)),
        	        ('SGD-hu-l1-1', SGDClassifier(loss='huber', penalty='l1', alpha=10.0, max_iter=IT)),
                    ('SGD-ei-l1-1', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=10.0, max_iter=IT)),
                    ('SGD-sei-l1-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=10.0, max_iter=IT)),
                    ('SGD-sl-l2-1', SGDClassifier(loss='squared_loss', penalty='l2', alpha=10.0, max_iter=IT)),
        	        ('SGD-hu-l2-1', SGDClassifier(loss='huber', penalty='l2', alpha=10.0, max_iter=IT)),
                    ('SGD-ei-l2-1', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=10.0, max_iter=IT)),
                    ('SGD-sei-l2-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=10.0, max_iter=IT)),
                    ('SGD-sl-e-1', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=10.0, max_iter=IT)),
        	        ('SGD-hu-e-1', SGDClassifier(loss='huber', penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('SGD-ei-e-1', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('SGD-sei-e-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('SGD-sl-n-1', SGDClassifier(loss='squared_loss', penalty='none', alpha=10.0, max_iter=IT)),
        	        ('SGD-hu-n-1', SGDClassifier(loss='huber', penalty='none', alpha=10.0, max_iter=IT)),
                    ('SGD-ei-n-1', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=10.0, max_iter=IT)),
                    ('SGD-sei-n-1', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=10.0, max_iter=IT)),

                    ('SGD-sl-l1-2', SGDClassifier(loss='squared_loss', penalty='l1', alpha=100.0, max_iter=IT)),
        	        ('SGD-hu-l1-2', SGDClassifier(loss='huber', penalty='l1', alpha=100.0, max_iter=IT)),
                    ('SGD-ei-l1-2', SGDClassifier(loss='epsilon_insensitive', penalty='l1', alpha=100.0, max_iter=IT)),
                    ('SGD-sei-l1-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l1', alpha=100.0, max_iter=IT)),
                    ('SGD-sl-l2-2', SGDClassifier(loss='squared_loss', penalty='l2', alpha=100.0, max_iter=IT)),
        	        ('SGD-hu-l2-2', SGDClassifier(loss='huber', penalty='l2', alpha=100.0, max_iter=IT)),
                    ('SGD-ei-l2-2', SGDClassifier(loss='epsilon_insensitive', penalty='l2', alpha=100.0, max_iter=IT)),
                    ('SGD-sei-l2-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='l2', alpha=100.0, max_iter=IT)),
                    ('SGD-sl-e-2', SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=100.0, max_iter=IT)),
        	        ('SGD-hu-e-2', SGDClassifier(loss='huber', penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('SGD-ei-e-2', SGDClassifier(loss='epsilon_insensitive', penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('SGD-sei-e-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('SGD-sl-n-2', SGDClassifier(loss='squared_loss', penalty='none', alpha=100.0, max_iter=IT)),
        	        ('SGD-hu-n-2', SGDClassifier(loss='huber', penalty='none', alpha=100.0, max_iter=IT)),
                    ('SGD-ei-n-2', SGDClassifier(loss='epsilon_insensitive', penalty='none', alpha=100.0, max_iter=IT)),
                    ('SGD-sei-n-2', SGDClassifier(loss='squared_epsilon_insensitive', penalty='none', alpha=100.0, max_iter=IT)),

                    #  Perceptron 
                    ('Perceptron-l1--4', Perceptron(penalty='l1', max_iter=IT)),
                    ('Perceptron-l2--4', Perceptron(penalty='l2', max_iter=IT)),
                    ('Perceptron-e--4', Perceptron(penalty='elasticnet', max_iter=IT)),
                    ('Perceptron-n--4', Perceptron(penalty='none', max_iter=IT)),

                    ('Perceptron-l1--6', Perceptron(penalty='l1', alpha=0.000001, max_iter=IT)),
                    ('Perceptron-l2--6', Perceptron(penalty='l2', alpha=0.000001, max_iter=IT)),
                    ('Perceptron-e--6', Perceptron(penalty='elasticnet', alpha=0.000001, max_iter=IT)),
                    ('Perceptron-n--6', Perceptron(penalty='none', alpha=0.000001, max_iter=IT)),

                    ('Perceptron-l1--2', Perceptron(penalty='l1', alpha=0.01, max_iter=IT)),
                    ('Perceptron-l2--2', Perceptron(penalty='l2', alpha=0.01, max_iter=IT)),
                    ('Perceptron-e--2', Perceptron(penalty='elasticnet', alpha=0.01, max_iter=IT)),
                    ('Perceptron-n--2', Perceptron(penalty='none', alpha=0.01, max_iter=IT)),

                    ('Perceptron-l1--1', Perceptron(penalty='l1', alpha=0.1, max_iter=IT)),
                    ('Perceptron-l2--1', Perceptron(penalty='l2', alpha=0.1, max_iter=IT)),
                    ('Perceptron-e--1', Perceptron(penalty='elasticnet', alpha=0.1, max_iter=IT)),
                    ('Perceptron-n--1', Perceptron(penalty='none', alpha=0.1, max_iter=IT)),

                    ('Perceptron-l1--0', Perceptron(penalty='l1', alpha=1.0, max_iter=IT)),
                    ('Perceptron-l2--0', Perceptron(penalty='l2', alpha=1.0, max_iter=IT)),
                    ('Perceptron-e--0', Perceptron(penalty='elasticnet', alpha=1.0, max_iter=IT)),
                    ('Perceptron-n--0', Perceptron(penalty='none', alpha=1.0, max_iter=IT)),

                    ('Perceptron-l1-1', Perceptron(penalty='l1', alpha=10.0, max_iter=IT)),
                    ('Perceptron-l2-1', Perceptron(penalty='l2', alpha=10.0, max_iter=IT)),
                    ('Perceptron-e-1', Perceptron(penalty='elasticnet', alpha=10.0, max_iter=IT)),
                    ('Perceptron-n-1', Perceptron(penalty='none', alpha=10.0, max_iter=IT)),

                    ('Perceptron-l1-2', Perceptron(penalty='l1', alpha=100.0, max_iter=IT)),
                    ('Perceptron-l2-2', Perceptron(penalty='l2', alpha=100.0, max_iter=IT)),
                    ('Perceptron-e-2', Perceptron(penalty='elasticnet', alpha=100.0, max_iter=IT)),
                    ('Perceptron-n-2', Perceptron(penalty='none', alpha=100.0, max_iter=IT)),



                    # Passive aggressive 
                    # TODO: what is step size and maybe vary
                    ('PA-I-0', PassiveAggressiveClassifier(loss='hinge', max_iter=IT)),
                    ('PA-II-0', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT)),

                    ('PA-I--1', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.1)),
                    ('PA-II--1', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.1)),

                    ('PA-I--2', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.01)),
                    ('PA-II--2', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.01)),

                    ('PA-I--3', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.001)),
                    ('PA-II--3', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.001)),

                    ('PA-I--4', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.0001)),
                    ('PA-II--4', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.0001)),

                    ('PA-I-1', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=10.0)),
                    ('PA-II-1', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=10.0)),

                    ('PA-I-2', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=100.0)),
                    ('PA-II-2', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=100.0)),


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
                    ("BNB-0.5", BernoulliNB(alpha=0.25)),
                    ("BNB-0.25", BernoulliNB(alpha=0.25)),
                    ("BNB-0.0", BernoulliNB(alpha=0.0)),

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
    
    print(len(features))
    print(len(features[0]))
    print(len(targets))
    print(len(trainSetsF))
    print(len(testSetsF))
    print(len(trainSetsF[1]))
    print(len(testSetsF[1]))

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
    for scaler in scalers:

        #for dimReducer in dimReduction:



        for classifier in classifiers:

            start = time.time()

            a = []

            for trainF, trainT, testF, testT in zip(trainSetsF, trainSetsT, testSetsF, testSetsT):
                
                m = 0

                if (not(scaler is None)):
                    trainF = scaler[1].fit_transform(trainF)
                    testF = scaler[1].transform(testF)

                # try to partial fit everything

                # TODO: change to fit, because to slow, maybe it helps
                for caseF, caseT in zip(trainF, trainT):
                    caseT = np.array([caseT])
                    classifier[1].partial_fit(X=caseF.reshape(1,-1), y=caseT, classes=np.array([0,1,2,3]))

                # I wont use score here, because it returns a harsh
                # metric which will always be low.
                # I will do it manually

                pred = classifier[1].predict(testF)

                for p,t in zip(pred, testT):
                    if (p == t):
                        m += 1
                m = m / len(testT)
                a.append(m)

            me = np.mean(np.array(a))
            va = np.std(np.array(a))

            sec = time.time()-start

            cNames.append(classifier[0])
            sNames.append(scaler[0])
            times.append(sec)
            scores.append(a)
            means.append(me)
            stds.append(va)
            scAndCl.append((scaler[1],classifier[1]))

            z += 1

            ms = np.mean(np.array(times))

            print(str(z)+"/"+str(size)+" completed. This: "+str(sec)+" sec. Average: "+str(ms)+" sec.       ", end='\r')

    print("Completed analysis. \n Constructing data about top 6.")

    top6 = [-1,-1,-1,-1,-1,-1]

    for i in range(0, len(means)):
        j = 0
        while(( top6[j] == -1 or means[i] >= means[top6[j]]) and j < 6):
            j += 1
        if (j != 0):
            k = j - 1
            h = 1
            while( k >= h):
                top6[h-1] = top6[h]
                h += 1
            top6[k] = i

    gaps = np.arange(0,100,10)
    vals = []
    for i in range(0,len(top6)):
        scal, clas = scAndCl[top6[i]]
        
        val = []
        
        for j in gaps:
            fs = scal.fit_transform(features[0:NB_ADAPTION_PER_CYCLE*j-1])
            ts = scal.transform(features[NB_ADAPTION_PER_CYCLE*j:])
            clas.fit(fs, targets[0:NB_ADAPTION_PER_CYCLE*j-1])
            pred = clas.predict(ts)

            q = 0
            for p,t in zip(pred, targets[NB_ADAPTION_PER_CYCLE*j:]):
                if (p == t):
                    q +=1
            val.append(1-(q/len(pred)))
        vals.append(np.array(val))

    fig = plt.figure()
    ids = np.arange(0,kf,1)
    for i in range(len(top6) -1 ,-1,-1):
        plt.subplot(6,2,2*(6-i))
        plt.bar(ids, scores[top6[i]][ids], label="Scores over all folds")
        plt.axhline(means[top6[i]], 'r', label="Mean")
        plt.ylabel("1-error rate")
        plt.xlabel("fold")
        plt.title(str(i)+") "+str(cNames[top6[i]])+"\n"+str(sNames[top6[i]])+"\n Mean: "+str(means[top6[i]])+"\n std: "+str(stds[top6[i]])+"\n Time: "+str(times[top6[i]])+" secs")

        plt.subplot(6,2, 2*(6-i)+1)
        plt.plot(gaps, vals[i])
        plt.ylabel("error rate")
        plt.xlabel("Nb of adaptation cycles")
        plt.title("Error conversion")

    fig.savefig('modelSelection.eps', format='eps', dpi=1200)
    fig.savefig('modelSelection.svg', format='svg', dpi=1200)
    fig.savefig('modelSelection.png', format='png', dpi=1200)


                
    data = {}
    data["cName"] = cNames
    data["sName"] = sNames
    data["time"] = times
    data["score"] = scores
    data["mean"] = means
    data["std"] = stds


    with open("modelSelectionResults.json","w") as f:
        json.dump(data, f)



                

                


    #if scaler is None:


main()
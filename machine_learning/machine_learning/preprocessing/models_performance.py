import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron, SGDRegressor
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from parser import load_data

classifiers = [
    ('SGD(default)', SGDClassifier()),
    ('SGD(log)', SGDClassifier(loss='log')),
    ('SGD(modified_huber)', SGDClassifier(loss='modified_huber')),
    ('SGD(perceptron)', SGDClassifier(loss='perceptron')),
    ('ASGD', SGDClassifier(average=True)),
    ('ASGD(log)', SGDClassifier(average=True, loss='log')),
    ('ASGD(modified_huber)', SGDClassifier(average=True, loss='modified_huber')),
    ('ASGD(squared_hinge)', SGDClassifier(average=True, loss='squared_hinge')),
    ('ASGD(perceptron)', SGDClassifier(average=True, loss='perceptron')),
    ('Perceptron(default)', Perceptron()),
    ('Perceptron(l2)', Perceptron(penalty='l2')),
    ('Perceptron(l1)', Perceptron(penalty='l1')),
    ('Perceptron(elasticnet)', Perceptron(penalty='elasticnet')),
    ('PA(default)', PassiveAggressiveClassifier()),
    ('PA(hinge)', PassiveAggressiveClassifier(loss='hinge')),
    ('PA(squared_hinge)', PassiveAggressiveClassifier(loss='squared_hinge')),
    ('APA(hinge)', PassiveAggressiveClassifier(average=True, loss='hinge')),
    ('APA(squared_hinge)', PassiveAggressiveClassifier(average=True, loss='squared_hinge'))
]

regressors = [
    ('SGD(default)', SGDRegressor()),
    ('SGD(huber)', SGDRegressor(loss='huber')),
    ('SGD(epsilon_insensitive)', SGDRegressor(loss='epsilon_insensitive')),
    ('SGD(squared_epsilon_insensitive)', SGDRegressor(loss='squared_epsilon_insensitive')),
    ('ASGD', SGDRegressor(average=True)),
    ('ASGD(huber)', SGDRegressor(average=True, loss='huber')),
    ('ASGD(epsilon_insensitive)', SGDRegressor(average=True, loss='epsilon_insensitive')),
    ('ASGD(squared_epsilon_insensitive)', SGDRegressor(average=True, loss='squared_epsilon_insensitive')),
    ('PA(default)', PassiveAggressiveRegressor()),
    ('PA(epsilon_insensitive)', PassiveAggressiveRegressor(loss='epsilon_insensitive')),
    ('PA(squared_epsilon_insensitive)', PassiveAggressiveRegressor(loss='squared_epsilon_insensitive')),
    ('APA(epsilon_insensitive)', PassiveAggressiveRegressor(average=True, loss='epsilon_insensitive')),
    ('APA(squared_epsilon_insensitive)', PassiveAggressiveRegressor(average=True, loss='squared_epsilon_insensitive'))
]

DIR_PATH = 'machine_learning/results/'
TESTING_SIZE = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.01]
ROUNDS = 1

features, classification_target, regression_target = load_data(features_type='selected')

for target, estimators, target_type in zip([classification_target, regression_target], [classifiers, regressors],
                                           ['classification', 'regression']):

    x_axis = 1. - np.array(TESTING_SIZE)

    plt.figure(figsize=(20, 20))
    fig_index = 0

    for scaler in [None, MinMaxScaler, StandardScaler, MaxAbsScaler]:
        estimator_index = 0
        title = 'Without Scaler'

        if scaler != None:
            title = ' With ' + scaler.__name__ 

        for name, clf in estimators:
            rng = np.random.RandomState(42)
            y_axis = []

            for size in TESTING_SIZE:
                y_axis_mean = []

                for _ in range(ROUNDS):
                    training_features, testing_features, training_target, testing_target = train_test_split(
                        features, target, test_size=size, random_state=rng)

                    if scaler != None:
                        m_scaler = scaler()
                        m_scaler.partial_fit(training_features)
                        training_features = m_scaler.transform(training_features)
                        testing_features = m_scaler.transform(testing_features)

                    if target_type == 'classification':
                        clf.partial_fit(training_features, training_target, classes=np.array([0, 1]))
                        predictions = clf.predict(testing_features)
                    else:
                        clf.partial_fit(training_features, training_target)
                        predictions = clf.predict(testing_features)
                        for i in range(0, len(testing_target)):
                            if predictions[i] < 10 and testing_target[i] < 10 or predictions[i] >= 10 and testing_target[i] >= 10:
                                predictions[i] = 1
                                testing_target[i] = 1
                            else:
                                predictions[i] = 0
                                testing_target[i] = 1

                    y_axis_mean.append(1 - np.mean(predictions == testing_target))

                y_axis.append(np.mean(y_axis_mean))

            if estimator_index % 5 == 0:
                fig_index += 1
                
                if target_type == 'classification':
                    plt.subplot(4, 4, fig_index)
                else:
                    plt.subplot(4, 3, fig_index)
                
                plt.title(title)

            plt.plot(x_axis, y_axis, label=name)
            plt.legend(loc='upper right')
            plt.xlabel("Training Size")
            plt.ylabel("Test Error Rate")
            estimator_index += 1
    
    plt.subplots_adjust(top=0.98, bottom=0.03, left=0.04, right=0.98, hspace=0.25, wspace=0.35)
    plt.savefig(DIR_PATH + target_type + '_models_performance.pdf')
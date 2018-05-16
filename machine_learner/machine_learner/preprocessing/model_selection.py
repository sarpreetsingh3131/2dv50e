import random, numpy as np, json, matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

penalty = ['l1', 'l2', 'elasticnet'][0]

sgd_classifiers = [
    ('SGD-h', SGDClassifier(loss='hinge', penalty=penalty)),
    ('SGD-l', SGDClassifier(loss='log', penalty=penalty)),
    ('SGD-mh', SGDClassifier(loss='modified_huber', penalty=penalty)),
    ('SGD-sh', SGDClassifier(loss='squared_hinge', penalty=penalty)),
    ('SGD-p', SGDClassifier(loss='perceptron', penalty=penalty))
]

sgd_regressors = [
    ('SGD-sl', SGDRegressor(loss='squared_loss', penalty=penalty)),
    ('SGD-h', SGDRegressor(loss='huber', penalty=penalty)),
    ('SGD-ei', SGDRegressor(loss='epsilon_insensitive', penalty=penalty)),
    ('SGD-sei', SGDRegressor(loss='squared_epsilon_insensitive', penalty=penalty))
]

selected_classifiers = [
    ('SGD', SGDClassifier(loss='hinge', penalty='l1')),
    ('Perceptron-I', Perceptron(penalty='l1')),
    ('Perceptron-II', Perceptron(penalty='l2')),
    ('Perceptron-II', Perceptron(penalty='elasticnet')),
    ('PA-I', PassiveAggressiveClassifier(loss='hinge')),
    ('PA-II', PassiveAggressiveClassifier(loss='squared_hinge'))
]

selected_regressors = [
    ('SGD-l1', SGDRegressor(loss='squared_loss', penalty='l2')),
    ('PA-I', PassiveAggressiveRegressor(loss='epsilon_insensitive')),
    ('PA-II', PassiveAggressiveRegressor(loss='squared_epsilon_insensitive'))
]

scalers = [
    ('Min-Max', MinMaxScaler()),
    ('Standarization', StandardScaler()),
    ('Max-Abs', MaxAbsScaler())
]

data = json.load(open('machine_learner/collected_data/dataset_with_selected_features.json'))
features = data['features']
training_cycles = [15, 30, 45, 60, 70]
rounds = 20

for target, target_type, models in zip(
                                        [ data['classification_target'], data['regression_target']],
                                        ['Classification', 'Regression'],
                                        [selected_classifiers, selected_regressors]
                                    ):
    plt.figure()
    plt_index = 1
    for scaler_name, scaler in scalers:
        plt.subplot(1, 3, plt_index)
        plt_index += 1        
        for model_name, model in models:
            error_rate = []
            for cycle in training_cycles:
                scores = []
                for i in range(rounds):
                    training_features = features[0: 216 * cycle]
                    testing_features = features[216 * cycle:]
                    training_target = target[0: 216 * cycle]
                    testing_target = target[216 * cycle:]
                    scaler.fit(training_features)
                    training_features = scaler.transform(training_features)
                    testing_features = scaler.transform(testing_features)
                    model.fit(training_features, training_target)
                    predictions = model.predict(testing_features)
                    if target_type == 'Regression':
                        for i in range(0, len(testing_target)):
                            if predictions[i] < 10.0 and testing_target[i] < 10.0 or \
                               predictions[i] >= 10.0 and testing_target[i] >= 10.0:
                                    predictions[i],  testing_target[i] = 1, 1
                            else:
                                predictions[i], testing_target[i] = 0, 1
                    scores.append(1 - np.mean(predictions == testing_target))
                error_rate.append(np.mean(scores))
            plt.plot(training_cycles, error_rate, label=model_name)
        plt.xlabel('Training Cycles')
        plt.ylabel('Testing Error Rate (%)')
        plt.xticks([15, 30, 45, 60])
        plt.title('With ' + scaler_name + ' Scaling')
        plt.grid()
        if plt_index == 3:
            plt.legend()
        if target_type == 'Classification':
            plt.ylim(ymax=0.16)
        else:
            plt.ylim(ymax=0.30)    
    plt.show()
    
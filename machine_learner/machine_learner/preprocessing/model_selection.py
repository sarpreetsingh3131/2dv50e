import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

classifiers = [
    ('SGD-I', SGDClassifier()), ('SGD-II', SGDClassifier(penalty='l1')),
    ('Perceptron-I', Perceptron(penalty='l1')), ('Perceptron-II', Perceptron(penalty='elasticnet')),
    ('PA-I', PassiveAggressiveClassifier()), ('PA-II', PassiveAggressiveClassifier(loss='squared_hinge'))
]
regressors = [
    ('SGD-I', SGDRegressor()), ('SGD-II', SGDRegressor(loss='huber')), ('SGD-III', SGDRegressor(loss='epsilon_insensitive')),
    ('PA-I', PassiveAggressiveRegressor()), ('PA-II', PassiveAggressiveRegressor(loss='squared_epsilon_insensitive'))
]


data = json.load(open('machine_learner/collected_data/dataset_with_selected_features.json'))
features = data['features']
test_sizes = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.01]
rounds = 20

for target, target_type, models in zip([data['classification_target'], data['regression_target']], ['Classification', 'Regression'],
                                        [classifiers, regressors]):
    
    plt.figure()
    plt_index = 1
    
    for scaler_name, scaler in [('Min-Max', MinMaxScaler), ('Standarization',StandardScaler), ('Max-Abs', MaxAbsScaler)]:
        plt.subplot(1, 3, plt_index)
        plt_index += 1        
        
        for model_name, model in models:
            accuracy = []
            
            for size in test_sizes:
                prediction_score = []
                
                for i in range(rounds):
                    training_features, testing_features, training_target, testing_target = train_test_split(features, target, test_size=size, random_state=5)
                    scaler_model = scaler()
                    scaler_model.partial_fit(training_features)
                    training_features = scaler_model.transform(training_features)
                    testing_features = scaler_model.transform(testing_features)
                    predictions = []
                    
                    if target_type == 'Classification':
                        model.partial_fit(training_features, training_target, classes=np.array([0, 1]))
                        predictions = model.predict(testing_features)
                    else:
                        model.partial_fit(training_features, training_target)
                        predictions = model.predict(testing_features)
                        
                        for i in range(0, len(testing_target)):
                            if predictions[i] < 10 and testing_target[i] < 10 or predictions[i] >= 10 and testing_target[i] >= 10:
                                predictions[i], testing_target[i] =  1, 1
                            else:
                                predictions[i], testing_target[i] = 0, 1

                    prediction_score.append(1 - np.mean(predictions == testing_target))
                accuracy.append(np.mean(prediction_score))
            plt.plot(1. - np.array(test_sizes), accuracy, label=model_name)
        plt.xlabel('Training Size (%)', fontsize=8)
        plt.ylabel('Test Error Rate (%)', fontsize=8)
        plt.xticks(fontsize=8)
        plt.title('With ' + scaler_name + ' Scaling', fontsize=8)

        if plt_index == 4:
            plt.legend()

    plt.show()

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from parser import load_data
import matplotlib.pyplot as plt

DIR_PATH = 'machine_learning/results/'

features, classification_target, regression_target = load_data(features_type='all')
for target_type, target, model in zip(['classification', 'regression'],[classification_target, regression_target],
                                    [ExtraTreesClassifier, ExtraTreesRegressor]):
    model = model()
    model.fit(features, target)
    importances = model.feature_importances_

    plt.figure()
    plt.bar(range(0, 17), importances[:17], color='r', align='center', label='0-16 SNR')
    plt.bar(range(17, 34), importances[17:34], color='b', align='center', label='17-33 Distributions')
    plt.bar(range(34, 48), importances[34:], color='green', align='center', label='33-47 Traffic')

    plt.xlabel('Features')
    plt.ylabel('Importances')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.savefig(DIR_PATH + target_type + '_feature_importances.pdf')

from sklearn.feature_selection import SelectKBest, f_classif, f_regression, SelectPercentile, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import numpy as np
from dataset import load_data
import matplotlib.pyplot as plt

DIR_PATH = 'machine_learning/results/'


"""
def feature_scores(features, target, description, score_functions):
    plt.figure(figsize=(15, 8))
    for index, score_fun in enumerate(score_functions):
        fit = score_fun.fit(features, target)
        scores = np.array(fit.scores_, dtype='float')
        y_axis = np.arange(len(scores))
        title = 'Feature Scores For ' + description[index]

        plt.subplot(1, 2, index + 1)

        for s, e, c, l in zip([0, 17, 33], [17, 34, 48], ['r', 'b', 'g'], ['SNR',  'Distribution', 'Traffic']):
            plt.bar(y_axis[s:e], scores[s:e], align='center', color=c, label=l)

        plt.xlabel('Feature')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc='upper right')
    plt.savefig(DIR_PATH + title + '.pdf')
    print('graph saved')
"""


def feature_importances(features, target, target_type, clf):
    forest = clf(n_estimators=250, random_state=42)
    forest.fit(features, target)
    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.bar(range(features.shape[1]), importances[indices], color='r', align='center')
    plt.xlim([-1, features.shape[1]])

    title = 'Feature Importances For ' + target_type + ' By ' + clf.__name__
    plt.suptitle(title)
    plt.savefig(DIR_PATH + title + '.pdf')
    print(importances, 'graph saved')


features, classification_target, regression_target = load_data(features_type='all')

for target_type, target, clf in zip(['Classification', 'Regression'],
                                    [classification_target, regression_target],
                                    [ExtraTreesClassifier, ExtraTreesRegressor]):

    feature_importances(features, target, target_type, clf)

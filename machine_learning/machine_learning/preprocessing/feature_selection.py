from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from dataset import load_data
import matplotlib.pyplot as plt

DIR_PATH = 'machine_learning/preprocessing/results/'


def feature_scores(features, target, target_type, score_function):
    k_best = SelectKBest(score_func=score_function, k='all')
    fit = k_best.fit(features, target)
    scores = fit.scores_

    plt.figure()
    plt.bar(np.arange(len(scores)), scores, align='center', alpha=1)
    plt.xticks([10, 20, 24], ['SNR', 'Distribution', 'Traffic'])

    title = 'Feature Scores For ' + target_type + ' By ' + score_function.__name__
    plt.suptitle(title)
    plt.savefig(DIR_PATH + title)


def feature_importances(features, target, target_type):
    forest = ExtraTreesClassifier(n_estimators=25, random_state=0)
    forest.fit(features, target)
    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.bar(range(features.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
    #plt.xticks(range(features.shape[1]), indices)
    plt.xticks([10, 20, 24], ['SNR', 'Distribution', 'Traffic'])
    plt.xlim([-1, features.shape[1]])

    title = 'Feature Importances For ' + target_type + ' By Extra Tress Classifier'
    plt.suptitle(title)
    plt.savefig(DIR_PATH + title + '.pdf')


features, classification_target, regression_target = load_data()
for target_type, target, score_func in zip(
        ['Classification', 'Regression'], [classification_target, regression_target], [f_classif, f_regression]):

    feature_importances(features, target, target_type)
    feature_scores(features, target, target_type, score_func)

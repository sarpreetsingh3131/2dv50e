from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from parser import load_data
import matplotlib.pyplot as plt

DIR_PATH = 'machine_learning/results/'


features, classification_target, regression_target = load_data(features_type='all')

plt.figure(figsize=(15, 10))
fig_index = 1

for target_type, target, clf in zip(['Classification', 'Regression'],
                                    [classification_target, regression_target],
                                    [ExtraTreesClassifier, ExtraTreesRegressor]):

    forest = clf(n_estimators=250, random_state=42)
    forest.fit(features, target)
    importances = forest.feature_importances_

    plt.subplot(2, 1, fig_index)
    plt.bar(range(0, 17), importances[:17], color='r', align='center', label='0-16 SNR')
    plt.bar(range(17, 34), importances[17:34], color='b', align='center', label='17-33 Distributions')
    plt.bar(range(34, 48), importances[34:], color='orange', align='center', label='33-47 Traffic')

    plt.xlabel('Features')
    plt.ylabel('Score')
    plt.title('Feature Importances For ' + target_type)
    plt.legend()
    fig_index += 1

plt.savefig(DIR_PATH + 'Feature Selection.pdf')
print('graph saved')

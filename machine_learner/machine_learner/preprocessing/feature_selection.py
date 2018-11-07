import matplotlib.pyplot as plt
import json
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import os

filep = os.path.realpath(__file__)
parent = os.path.dirname(filep)
grandpa = os.path.dirname(parent)
path = os.path.abspath(os.path.join(grandpa, "collected_data", "dataset_with_all_features.json"))


data = json.load(open(path))

plt_index = 1

classif = ("Classification", data['classification_target'], ExtraTreesClassifier)
regres = ("Regression", data["regression_target"], ExtraTreesRegressor)

for target_type, target, model in [classif, regres]:

    model = model(random_state=10)
    model.fit(data['features'], target)

    importances = model.feature_importances_

    plt.subplot(1, 2, plt_index)

    plt.bar(range(1, 18), importances[:17], color='r', label='1-17 SNR')
    plt.bar(range(18, 35), importances[17:34],
            color='b', label='18-34 Packets Distribution')
    plt.bar(range(35, 49), importances[34:],
            color='green', label='35-48 Traffic Load')

    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(target_type)
    plt.xticks([1, 17, 34, 48])

    plt_index += 1

    if plt_index == 2:
        plt.legend()

plt.tight_layout()
plt.show()

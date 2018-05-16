import matplotlib.pyplot as plt, json
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

data = json.load(open('machine_learner/collected_data/dataset_with_all_features.json'))
plt_index = 1

for target_type, target, model in zip(
                                    ['Classification', 'Regression'],
                                    [data['classification_target'], data['regression_target']],
                                    [ExtraTreesClassifier, ExtraTreesRegressor]
                                ):
    model = model(random_state=10)
    model.fit(data['features'], target)
    importances = model.feature_importances_
    plt.subplot(1, 2, plt_index)
    plt.bar(range(1, 18), importances[:17], color='r', label='1-17 SNR')
    plt.bar(range(18, 35), importances[17:34], color='b', label='18-34 Packets Distribution')
    plt.bar(range(35, 49), importances[34:], color='green',label='35-48 Traffic Load')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(target_type)
    plt.xticks([1, 17, 34, 48])
    plt_index +=1
    if plt_index == 2:
        plt.legend()
plt.show()

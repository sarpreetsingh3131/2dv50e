import matplotlib.pyplot as plt
import json
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import os

def doFeatureSelection():
    path = os.path.join('machine_learner', 'collected_data', 'dataset_with_all_features.json')

    data = json.load(open(path))

    plt_index = 1

    classif = ('Classification', [data['target_classification_packetloss'], data['target_classification_latency']], ExtraTreesClassifier)
    # NOTE: add regression later
    # regres = ('Regression', [data['target_regression_packetloss'], data['target_regression_latency']], ExtraTreesRegressor)

    for target_type, targets, model in [classif]:
        target_packetloss, target_latency = targets

        totalTargets = []
        # TODO adjust this later on for regression
        for target_pl, target_l in zip(target_packetloss, target_latency):
            totalTargets.append(target_pl + (target_l << 1))

        model = model(random_state=10)
        model.fit(data['features'], totalTargets)

        importances = model.feature_importances_

        plt.subplot(1, 1, plt_index)

        plt.bar(range(1, 18), importances[:17], color='r', label='1-17 SNR')
        plt.bar(range(18, 35), importances[17:34], color='b', label='18-34 Power Settings')
        plt.bar(range(35, 52), importances[34:51], color='orange', label='35-51 Packets Distribution')
        plt.bar(range(52, 66), importances[51:65], color='green', label='52-65 Traffic Load')

        print("SNR:   " + str([f'{i:.4f}' for i in importances[:17]]))
        print("Power: " + str([f'{i:.4f}' for i in importances[17:34]]))
        print("Distr: " + str([f'{i:.4f}' for i in importances[34:51]]))
        print("Load:  " + str([f'{i:.4f}' for i in importances[51:65]]))

        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(target_type)
        plt.xticks([1, 17, 34, 51, 65])

        plt_index += 1

        if plt_index == 2:
            plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    doFeatureSelection()


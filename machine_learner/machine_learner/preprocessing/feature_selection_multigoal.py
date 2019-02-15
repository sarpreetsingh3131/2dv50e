import matplotlib.pyplot as plt
import json
import sys
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import os

def doFeatureSelectionClass(data):
    target_packetloss, target_latency = data['target_classification_packetloss'], data['target_classification_latency']

    totalTargets = []
    for target_pl, target_l in zip(target_packetloss, target_latency):
        totalTargets.append(target_pl + (target_l << 1))

    model = ExtraTreesClassifier(random_state=10)
    model.fit(data['features'], totalTargets)

    importances = model.feature_importances_
    return importances


def doFeatureSelectionRegression(data):
    target_packetloss, target_latency = data['target_regression_packetloss'], data['target_regression_latency']

    model1 = ExtraTreesRegressor(random_state=10)
    model1.fit(data['features'], target_packetloss)

    model2 = ExtraTreesRegressor(random_state=10)
    model2.fit(data['features'], target_latency)

    importances1 = model1.feature_importances_
    importances2 = model2.feature_importances_

    importances = [(importances1[i] + importances2[i]) / 2 for i in range(len(importances1))]
    return importances


def doFeatureSelection(network):
    path = os.path.join('machine_learner', 'collected_data', 'dataset_with_all_features.json')
    data = json.load(open(path))

    importancesClass = doFeatureSelectionClass(data)
    importancesRegr = doFeatureSelectionRegression(data)

    importances = [(importancesClass[i] + importancesRegr[i]) / 2 for i in range(len(importancesClass))]

    plt.subplot(1, 1, 1)

    if network == 'DeltaIoTv1':
        plt.bar(range(1, 18), importances[:17], color='r', label='1-17 SNR')
        plt.bar(range(18, 35), importances[17:34], color='b', label='18-34 Power Settings')
        plt.bar(range(35, 52), importances[34:51], color='orange', label='35-51 Packets Distribution')
        plt.bar(range(52, 66), importances[51:65], color='green', label='52-65 Traffic Load')

        plt.xticks([1, 17, 34, 51, 65])

    elif network == 'DeltaIoTv2':
        plt.bar(range(1, 43), importances[:42], color='r', label='1-42 SNR')
        plt.bar(range(43, 85), importances[42:84], color='b', label='43-84 Power Settings')
        plt.bar(range(85, 127), importances[84:126], color='orange', label='85-126 Packets Distribution')
        plt.bar(range(127, 163), importances[126:162], color='green', label='127-162 Traffic Load')

        plt.xticks([1, 42, 84, 126, 162])

    else:
        print(f'Network \'{network}\' is currently not supported.')
        sys.exit(1)

    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature importances')

    plt.legend()


    indices = np.array([i for i in range(len(importances)) if importances[i] != 0])

    # save the selected features in a separate file
    newFileData = json.load(open(path))

    newFeatures = []
    for feature_vec in newFileData['features']:
        newFeatures.append(np.array(feature_vec)[indices].tolist())

    newFileData['features'] = newFeatures

    outputPath = os.path.join('machine_learner','collected_data',f'dataset_selected_features.json')
    with open(outputPath, 'w') as f:
        json.dump(newFileData, f, indent=1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # TODO verify
    doFeatureSelection('DeltaIoTv2')


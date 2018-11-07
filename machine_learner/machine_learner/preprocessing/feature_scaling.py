"""This file analyzes the different feature scaling algorithms

Yeah...
It analyzes 3 different scalers on the dataset with selected features.
So this is done after the feature selection.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

#data = json.load(
#    open('machine_learner/collected_data/dataset_with_selected_features.json'))

# You have to use this, because this file is executed by itself to analyze and
# is not used when you run the simulator
filep = os.path.realpath(__file__)
parent = os.path.dirname(filep)
grandpa = os.path.dirname(parent)
path = os.path.abspath(os.path.join(grandpa, "collected_data", "dataset_with_selected_features.json"))

data = json.load(open(path))
plt_index = 1
features = data['features']

scalers = [
    ('', None),
    ('Min-Max', MinMaxScaler),
    ('Standarization', StandardScaler),
    ('Max-Abs', MaxAbsScaler)
]

# Running through all the scalers
for name, scaler in scalers:

    plt.subplot(2, 2, plt_index)

    if scaler != None:

        model = scaler()

        model.partial_fit(data['features'])
        features = model.transform(data['features'])

        plt.title('With ' + name + ' Scaling')

    else:

        plt.title('Without Scaling')

    # You can use the features because they are the first ones.

    #TODO: hardcoded features and scale
    # plotting the SNR to the first 15 bars

    plt.bar(range(1, 18), features[0][:17], color='r', label='1-17 SNR')
    plt.bar(range(18, 24), features[0][17:23],
            color='b', label='18-23 Packets Distribution')
    plt.bar(range(24, 26), features[0][23:],
            color='green', label='24-25 Traffic Load')

    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.xticks([1, 9, 17, 25])

    plt_index += 1

    if scaler == None:
        plt.legend()

plt.tight_layout()
plt.show()

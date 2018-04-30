import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

data = json.load(open('machine_learner/collected_data/dataset_with_selected_features.json'))
plt_index = 1
features = data['features']

for name, scaler in [('', None), ('Min-Max', MinMaxScaler), ('Standarization', StandardScaler), ('Max-Abs', MaxAbsScaler)]:
    plt.subplot(2, 2, plt_index)
    
    if scaler != None:
        model = scaler()
        model.partial_fit(features)
        features = model.transform(features)
        plt.title('With ' + name + ' Scaling', fontsize=8)
    else:
        plt.title('Without Scaling', fontsize=8)
    
    plt.bar(range(1, 18), features[0][:17], color='r', label='SNR')
    plt.bar(range(18, 23), features[0][17:22], color='b', label='Distribution')
    plt.bar(range(23, 26), features[0][22:], color='green', label='Traffic')
    plt.xlabel('Features', fontsize=8)
    plt.ylabel('Value', fontsize=8)
    plt.xticks([1, 9, 17, 25])
    plt_index += 1
    
    if scaler == None:
        plt.legend(fontsize=8)

plt.show()

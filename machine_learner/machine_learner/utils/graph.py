from matplotlib import pyplot as plt
import numpy as np
import json
import matplotlib.patches as mpatches

def online_learning():
    plt.figure()
    data = {'classification': [], 'regression': [], 'activform': []}
    file_data = open('machine_learner/collected_data/online_learning.txt').readlines()
    file_data = [x.strip() for x in file_data]
    for line in file_data:
        content = line.split(';')
        if len(content) == 8:
            data['classification'].append(int(content[2]))
            data['regression'].append(int(content[3]))
            data['activform'].append(int(content[4]))

    for index, label in enumerate(['Classification', 'Regression']):
        plt.subplot(2, 1, index + 1)
        x_axis = np.arange(1, 176, 1)
        plt.xticks([1, 25, 50, 75, 100, 125, 150, 175])
        plt.plot(x_axis, data[str(label).lower()], label=['SGDClasifier', 'SGDRegressor'][index])
        plt.plot(x_axis, data['activform'], label='ActivFORM')
        plt.legend(fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
        plt.xlabel('Adaptation Cycles', fontsize=8)
        plt.ylabel('Adaptation Options', fontsize=8)

    plt.show()


def comparison():
    plt.figure()
    plt_index = 1
    for title in ['Energy Consumption (coulomb)', 'Packet Loss (%)', 'Adaptation Options', 'Adaptation Time (sec)']:
        data = {'activform': [], 'classification': [], 'regression': []}

        for file_name in ['activform', 'classification', 'regression']:
            file_data = open('machine_learner/collected_data/' + file_name + '.txt').readlines()
            file_data = [x.strip() for x in file_data]
            for line in file_data:
                content = line.split(';')

                if len(content) == 8:
                    if title == 'Energy Consumption (coulomb)':
                        data[file_name].append(float(content[6]))
                    
                    elif title == 'Packet Loss (%)':
                        data[file_name].append(float(content[5]))

                    elif title == 'Adaptation Options':
                        if file_name == 'activform':
                            data[file_name].append(216)
                        else:
                            data[file_name].append(int(content[3]))

                    elif title == 'Adaptation Time (sec)':
                        data[file_name].append((float(content[7]) - float(content[1])) / 1000.0)
                        
        plt.subplot(2, 2, plt_index)
        plt.boxplot([data[x] for x in ['activform', 'classification', 'regression']], positions=[1, 2, 3],
                    labels=['ActivFORM', 'SGDClassifier', 'SGDRegressor'], widths=.3)

        plt.ylabel(title, fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt_index += 1

        if plt_index == 3:
            plt.ylim(ymax=24)
 
    plt.show()


def time_measurement():
    plt.figure()
    plt_index = 1
    for title in ['Initial Training Time (ms)', 'Prediction Time (ms)', 'Training Time After Prediction (ms)']:
        data = {'classification': [], 'regression': []}

        for file_name in ['classification', 'regression']:
            file_data = open('machine_learner/collected_data/' + file_name + '.txt').readlines()
            file_data = [x.strip() for x in file_data]
            for line in file_data:
                content = line.split(';')
                
                if len(content) == 6 and title == 'Initial Training Time (ms)':
                    data[file_name].append(int(content[2]))
                
                elif len(content) == 8:
                    data[file_name].append(int(content[4]))
                    
                        
        plt.subplot(1, 3, plt_index)
        plt.boxplot([data[x] for x in ['classification', 'regression']], positions=[1, 2],
                    labels=['SGDClassifier', 'SGDRegressor'], widths=.3)

        plt.ylabel(title, fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt_index += 1
 
    plt.show()


def selected_adaptation_options():
    plt.figure()
    plt_index = 1
    data = json.load(open('machine_learner/collected_data/selected_adaptation_options.json'))
    i = 100

    for model_name, target_type in zip(['SGDClassifier', 'SGDRegressor'], [data[i]['classification'], data[i]['regression']]):
        plt.subplot(1, 2, plt_index)
        
        for pl, ec, target in zip(data[i]['packetLoss'], data[i]['energyConsumption'], target_type):
            plt.scatter(pl, ec, color='blue')

            if model_name == 'SGDClassifier' and target == 1 or model_name == 'SGDRegressor' and target < 10.0:
                plt.scatter(pl, ec, color='orange')
    
        plt.plot([10, 10], [(min(data[i]['energyConsumption'])), (max(data[i]['energyConsumption']))], color='red')
        plt.xlabel('Packet Loss (%)')
        plt.ylabel('Energy Consumption (coulomb)')
        plt.legend(handles=[mpatches.Patch(color='orange', label=model_name), mpatches.Patch(color='blue', label='ActivFORM')])
        plt_index += 1 

    plt.show()


#online_learning()
#comparison()
#time_measurement()
selected_adaptation_options()
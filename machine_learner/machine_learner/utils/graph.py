import numpy as np, json, matplotlib.pyplot as plt, matplotlib.patches as mpatches

def online_learning():
    plt.figure()
    data = {'classification': [], 'regression': [], 'activform': []}
    file_data = open('machine_learner/collected_data/online_learning.txt').readlines()
    file_data = [x.strip() for x in file_data]
    for line in file_data:
        content = line.split(';')
        if len(content) == 6:
            data['activform'].append(int(content[2]))
        elif len(content) == 11:
            data['classification'].append(int(content[4]))
            data['regression'].append(int(content[5]))
            data['activform'].append(int(content[6]))
    for index, label in enumerate(['Classification', 'Regression']):
        plt.subplot(2, 1, index + 1)
        plt.plot(np.arange(1, len(data['activform']) + 1, 1), data['activform'], label='ActivFORM', color='orange')
        plt.plot(
            np.arange(31, len(data['activform']) + 1, 1), data[str(label).lower()],
            label = ['Classification', 'Regression'][index], color=['blue', 'green'][index]
        )
        plt.xlabel('Adaptation Cycles')
        plt.ylabel('Adaptation Options')
        plt.xticks([1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
        if index == 0:
            plt.legend(
                bbox_to_anchor = (0., 1.02, 1., .102), loc=3, ncol=3, mode='expand', borderaxespad=0., handles=[
                mpatches.Patch(color='orange', label='ActivFORM'), mpatches.Patch(color='blue', label='Classification'),
                mpatches.Patch(color='green', label='Regression')]
            )
    plt.show()


def comparison():
    plt.figure()
    plt_index = 1
    for title in ['Packet Loss (%)', 'Energy Consumption (coulomb)', 'Adaptation Options', 'Adaptation Time (sec)']:
        data = {'activform': [], 'classification': [], 'regression': []}
        for file_name in ['activform', 'classification', 'regression']:
            file_data = open('machine_learner/collected_data/' + file_name + '.txt').readlines()
            file_data = [x.strip() for x in file_data]
            for line in file_data:
                content = line.split(';')
                if title == 'Adaptation Options' and file_name == 'activform':
                    data[file_name].append(216)
                elif title == 'Adaptation Options' and file_name != 'activform' and len(content) == 6:
                    if int(content[3]) == 0:
                        data[file_name].append(216)
                    else:
                        data[file_name].append(int(content[3]))
                elif title == 'Adaptation Time (sec)' and file_name == 'activform' and len(content) == 11:
                    time = float(content[10]) - float(content[1])
                    time -=  int(content[2]) + int(content[3]) + int(content[7]) + int(content[8]) + int(content[9])
                    data[file_name].append(time / 1000.0)
                elif title == 'Adaptation Time (sec)' and file_name != 'activform' and len(content) == 6:
                    data[file_name].append((float(content[5]) - float(content[1])) / 1000.0)
                elif title == 'Packet Loss (%)' and len(content) == 2:
                    data[file_name].append(float(content[0]))
                elif title == 'Energy Consumption (coulomb)' and len(content) == 2:
                    data[file_name].append(float(content[1]))
        plt.subplot(2, 2, plt_index)
        #print(title, 'cls', str(np.mean(data['classification'])), 'reg', str(np.mean(data['regression'])), 'ac', str(np.mean(data['activform'])))
        plt.boxplot(
            [data[x] for x in['activform', 'classification', 'regression']],
            positions = [1, 2, 3], labels=['ActivFORM', 'Classification', 'Regresssion'], widths=.3
        )
        plt.ylabel(title)
        plt_index += 1
    plt.show()


def time_measurement():
    plt.figure()
    plt_index = 1
    for title in ['Initial Training Time (sec)', 'Prediction Time (sec)', 'Training Time After Prediction (sec)']:
        data = {'classification': [], 'regression': []}
        for file_name in ['classification', 'regression']:
            file_data = open('machine_learner/collected_data/' + file_name + '.txt').readlines()
            file_data = [x.strip() for x in file_data]
            for line in file_data:
                content = line.split(';')
                if len(content) == 4 and title == 'Initial Training Time (sec)':
                    data[file_name].append(int(content[2]) / 1000.0)
                elif len(content) == 6 and title == 'Prediction Time (sec)':
                    data[file_name].append(int(content[2]) / 1000.0)
                elif len(content) == 6 and title == 'Training Time After Prediction (sec)':
                    data[file_name].append(int(content[4]) / 1000.0)
        plt.subplot(1, 3, plt_index)
        plt.boxplot(
            [data[x] for x in['classification', 'regression']],
            positions = [1, 2], labels=['Classification', 'Regression'], widths=.3
        )
        plt.ylabel(title)
        plt_index += 1
    plt.show()


def selected_adaptation_options():
    data = json.load(open('machine_learner/collected_data/selected_adaptation_options.json'))
    for i in range(len(data)):
        plt_index = 1
        plt.figure()
        for model_name, target_type in zip(
                                            ['Classification', 'Regression'],
                                            [data[i]['classification'], data[i]['regression']]
                                        ):
            plt.subplot(1, 2, plt_index)
            for pl, ec, target in zip(data[i]['packetLoss'], data[i]['energyConsumption'], target_type):
                plt.scatter(pl, ec, color='orange')
                if model_name == 'Classification' and target == 1:
                    plt.scatter(pl, ec, color='blue')
                elif model_name == 'Regression' and target < 10.0:
                     plt.scatter(pl, ec, color='green')
            plt.plot([10, 10], [(min(data[i]['energyConsumption'])), (max(data[i]['energyConsumption']))], color='red')
            plt.xlabel('Packet Loss (%)')
            plt.ylabel('Energy Consumption (coulomb)')
            if plt_index == 1:
                plt.legend(
                    bbox_to_anchor = (0.3, 1.02, 1.6, .102), loc=3, ncol=3, mode='expand', borderaxespad=0., handles=[
                    mpatches.Patch(color='blue', label='Classification'), mpatches.Patch(color='orange', label='ActivFORM'),
                    mpatches.Patch(color='green', label='Regression')
                ])
            plt_index += 1 
        plt.show()


#online_learning()
comparison()
#time_measurement()
#selected_adaptation_options()

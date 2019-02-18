import numpy as np
import csv
import os
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mlp


def overview_of_selected_adaptation_options():
    plt.figure()
    data = {'classification': [], 'regression': [], 'activform': []}
    file_data = open(
        'machine_learner/collected_data/overview_of_selected_adaptation_options.txt').readlines()
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
        plt.plot(np.arange(1, len(data['activform']) + 1, 1),
                 data['activform'], label='ActivFORM', color='orange')
        plt.plot(
            np.arange(31, len(data['activform']) + 1, 1),
            data[str(label).lower()],
            label=['Classification', 'Regression'][index],
            color=['blue', 'green'][index]
        )
        plt.xlabel('Adaptation Cycles')
        plt.ylabel('Adaptation Space')
        plt.ylim(ymax=216)
        plt.xticks([1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
        if index == 0:
            plt.legend(
                bbox_to_anchor=(0., 1.02, 1., .102),
                loc=3,
                ncol=3,
                mode='expand',
                borderaxespad=0.,
                handles=[
                    mpatches.Patch(color='orange', label='ActivFORMS'),
                    mpatches.Patch(color='blue', label='Classification'),
                    mpatches.Patch(color='green', label='Regression')
                ]
            )
    plt.show()


def comparison():
    plt.figure()
    plt_index = 1
    for title in ['Packet Loss (%)', 'Energy Consumption (coulomb)', 'Adaptation Space', 'Adaptation Time (sec)']:
        data = {'activform': [], 'classification': [], 'regression': []}
        for file_name in ['activform', 'classification', 'regression']:
            file_data = open('machine_learner/collected_data/' +
                             file_name + '.txt').readlines()
            file_data = [x.strip() for x in file_data]
            for index, line in enumerate(file_data):
                content = line.split(';')
                if title == 'Adaptation Space' and file_name == 'activform' and len(content) == 11:
                    data[file_name].append(216)
                elif title == 'Adaptation Space' and file_name != 'activform' and len(content) == 6:
                    if int(content[3]) == 0:
                        data[file_name].append(216)
                    else:
                        data[file_name].append(int(content[3]))
                elif title == 'Adaptation Time (sec)' and file_name == 'activform' and len(content) == 11:
                    time = float(content[10]) - float(content[1])
                    time -= int(content[2]) + int(content[3]) + \
                        int(content[7]) + int(content[8]) + int(content[9])
                    data[file_name].append(time / 1000.0)
                elif title == 'Adaptation Time (sec)' and file_name != 'activform' and len(content) == 6:
                    data[file_name].append(
                        (float(content[5]) - float(content[1])) / 1000.0)
                elif title == 'Packet Loss (%)' and len(content) == 2 and index >= 330:
                    data[file_name].append(float(content[0]))
                elif title == 'Energy Consumption (coulomb)' and len(content) == 2 and index >= 330:
                    data[file_name].append(float(content[1]))
        plt.subplot(2, 2, plt_index)
        print(title, 'cls', str(np.mean(data['classification'])), 'reg', str(
            np.mean(data['regression'])), 'ac', str(np.mean(data['activform'])))
        boxplot = plt.boxplot(
            [data[x] for x in['activform', 'classification', 'regression']],
            positions=[1, 2, 3],
            widths=.5,
            labels=['ActivFORMS', 'Classification', 'Regresssion'],
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 2}
        )
        for index, box in enumerate(boxplot['boxes']):
            box.set(facecolor=['orange', 'blue', 'green'][index])
        plt.ylabel(title)
        plt_index += 1
    plt.show()


def time_measurement():
    plt.figure()
    plt_index = 1
    for title in ['Initial Training Time (sec)', 'Prediction Time (sec)', 'Training Time After Prediction (sec)']:
        data = {'classification': [], 'regression': []}
        for file_name in ['classification', 'regression']:
            file_data = open('machine_learner/collected_data/' +
                             file_name + '.txt').readlines()
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
        print(title, 'cls', str(np.mean(data['classification'])), 'reg', str(
            np.mean(data['regression'])))
        boxplot = plt.boxplot(
            [data[x] for x in['classification', 'regression']],
            positions=[1, 2],
            widths=.5,
            labels=['Classification', 'Regression'],
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 2}
        )
        for index, box in enumerate(boxplot['boxes']):
            box.set(facecolor=['blue', 'green'][index])
        plt.ylabel(title)
        plt_index += 1
    plt.show()


def selected_adaptation_options():
    data = json.load(
        open('machine_learner/collected_data/selected_adaptation_options.json'))
    for i in range(295,len(data)):
        plt_index = 1
        fig = plt.figure()
        fig.suptitle('Selected configurations in a single cycle', fontsize=16)
        for model_name, target_type in zip(
            ['Classification'],
            [data[i]['classification']]
        ):
            plt.subplot(1, 1, plt_index)
            for pl, ec, target in zip(data[i]['packetLoss'], data[i]['energyConsumption'], target_type):
                plt.scatter(pl, ec, color='orange')
                if model_name == 'Classification' and target == 1:
                    plt.scatter(pl, ec, color='blue')
                elif model_name == 'Regression' and target < 10.0:
                    plt.scatter(pl, ec, color='green')
            plt.plot([10, 10], [(min(data[i]['energyConsumption'])),
                                (max(data[i]['energyConsumption']))], color='red')
            plt.xlabel('Packet Loss (%)')
            plt.ylabel('Energy Consumption (coulomb)')
            if plt_index == 1:
                plt.legend(
                    bbox_to_anchor=(0, 1.02, 1, .102),
                    loc=3,
                    ncol=3,
                    mode='expand',
                    borderaxespad=0.,
                    handles=[
                        mpatches.Patch(color='orange', label='ActivFORMS'),
                        mpatches.Patch(color='blue', label='Classification'),
                        # mpatches.Patch(color='green', label='Regression')
                    ]
                )
            plt_index += 1
        plt.show()

        
        

def selected_adaptation_options_multigoal(highlightBest=True):
    mlp.rcParams['font.size'] = 14
    data = json.load(open(sys.argv[1]))
    
    # interrestingCyclesV1Class = [203,204,226,237,282,287,288]
    # interrestingCyclesV1Regr = [211,225,226,231,232,233,277,283,284,286]
    # interrestingCyclesV2Class = [211,214,219,225,236,247,260,268,273,275,297]
    # interrestingCyclesV2Regr = [224,228,231,236,239,246,249,268,280,282]
    # cycles = [210]
    for i in range(len(data)-5,len(data)):
        for _, target_type in zip(
            ['Classification'],
            [data[i]['adaptationOptions']['classificationBefore']]
        ):
            bestPointQualities = {'latency': -1, 'packetLoss' : -1, 'energyConsumption': max(data[i]['adaptationOptions']['energyConsumption']), 'predictedRight': False}
            for pl, ec, la, target in zip(data[i]['adaptationOptions']['packetLoss'], data[i]['adaptationOptions']['energyConsumption'], data[i]['adaptationOptions']['latency'], target_type):
                if pl < 10 and la < 5 and ec < bestPointQualities['energyConsumption']:
                    bestPointQualities['latency'] = la
                    bestPointQualities['packetLoss'] = pl
                    bestPointQualities['energyConsumption'] = ec
                    bestPointQualities['predictedRight'] = (target == (1 if pl < 10 else 0) + (2 if la < 5 else 0))


            verifiedConfigurations = data[i]['adaptationOptions']['verifiedConfigurations']

            def drawFigure(quality1, quality2, xlabel, ylabel, figName, firstLine, secondLine=None):
                plt.figure(figsize=(12,10))
                index = 0
                for d1, d2, target in zip(data[i]['adaptationOptions'][quality1], data[i]['adaptationOptions'][quality2], target_type):
                    if target == 3:
                        plt.scatter(d1, d2, color='blue', s=15)
                    elif verifiedConfigurations[index] == 1:
                        plt.scatter(d1, d2, color='xkcd:darkish green', s=45, marker='*')
                    else:
                        plt.scatter(d1, d2, color='orange', s=10)
                    # if index == 741:
                    #     plt.scatter(d1,d2,color='black',s=150, marker='*', zorder=100)
                    # else:
                    #     plt.scatter(d1,d2,color='orange', s=10)
                    index += 1
            

                if highlightBest:
                    plt.scatter(
                        bestPointQualities[quality1], 
                        bestPointQualities[quality2], 
                        color=('green' if bestPointQualities['predictedRight'] else 'red'), 
                        marker='*', s=150
                    )
                plt.plot(firstLine['bound1'],firstLine['bound2'], color='red')
                if secondLine != None:
                    plt.plot(secondLine['bound1'],secondLine['bound2'], color='red')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)

                legendHandles = [
                    (mpatches.Patch(color='Red', label='Best configuration (selected)') if bestPointQualities['predictedRight'] \
                        else mpatches.Patch(color='Red', label='Best configuration (missed)')),
                    mpatches.Patch(color='blue', label='Relevant options'),
                    mpatches.Patch(color='orange', label='Irrelevant options'),
                    mpatches.Patch(color='xkcd:darkish green', label='Explored options')
                ]

                plt.legend(
                    bbox_to_anchor=(0, 1.02, 1, 1),
                    loc=3,
                    ncol=4 if highlightBest else 3,
                    mode='expand',
                    borderaxespad=0.,
                    handles=legendHandles[(0 if highlightBest else 1):4]
                )

                plt.savefig(os.path.join(sys.argv[2], f'{figName}.pdf'), bbox_inches='tight')
                plt.close()


            drawFigure(
                'packetLoss', 
                'latency', 
                'Packet loss (%)', 
                'Latency (%)', 
                f'Cycle_{i}_PLLA', 
                {'bound1': [min(data[i]['adaptationOptions']['packetLoss']), max(data[i]['adaptationOptions']['packetLoss'])], 'bound2': [5,5]},
                secondLine={'bound1': [10,10], 'bound2': [(min(data[i]['adaptationOptions']['latency'])), (max(data[i]['adaptationOptions']['latency']))]}
            )
            drawFigure(
                'packetLoss', 
                'energyConsumption', 
                'Packet loss (%)', 
                'Energy consumption (Coulomb)', 
                f'Cycle_{i}_PLEC',
                {'bound1': [10,10], 'bound2': [(min(data[i]['adaptationOptions']['energyConsumption'])), (max(data[i]['adaptationOptions']['energyConsumption']))]}
            )
            drawFigure(
                'latency', 
                'energyConsumption', 
                'Latency (%)', 
                'Energy consumption (Coulomb)', 
                f'Cycle_{i}_LAEC', 
                {'bound1': [5,5], 'bound2': [(min(data[i]['adaptationOptions']['energyConsumption'])), (max(data[i]['adaptationOptions']['energyConsumption']))]}
            )
        


def boxplotsResults(ignoreTraining = False): 
    if len(sys.argv) != 3:
        print('Make sure to provide the data files as commandline argument.')
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    with open(sys.argv[2], 'r') as f:
        # Regression shares the same overall data as classification, just other predictions
        dataRegr = json.load(f)
    

    if ignoreTraining:
        amtTraining = min(i for i in range(len(data)) if data[i]['training'] == 'false')
        data = data[amtTraining:]
        dataRegr = dataRegr[amtTraining:]

    bestConfigurationsACTIVForms = []
    bestConfigurationsClassification = []
    bestConfigurationsRegression = []

    overallData = {'totalAmount':len(data[0]['adaptationOptions']['latency']), 'amountCL':[], 'amountRE':[]}
    # indicesGoodOptions = []
    
    for cycle in data:

        bestOptionAF = {'energyConsumption': max(cycle['adaptationOptions']['energyConsumption']), 'latency': -10, 'packetLoss': -10}
        bestOptionClass = {'energyConsumption': max(cycle['adaptationOptions']['energyConsumption']), 'latency': -10, 'packetLoss': -10}

        amtOptions = len(cycle['adaptationOptions']['classificationBefore'])

        # ACTIVForms
        indicesAF = [i for i in range(amtOptions) if cycle['adaptationOptions']['packetLoss'][i] < 10 and cycle['adaptationOptions']['latency'][i] < 5]
        # indicesGoodOptions.append(indicesAF.copy())
        # overallData['amountC3'].append(len(indicesAF))

        # In case there are no configurations which satisfy both goals, fall back to failsafe configuration (consider whole space)
        if len(indicesAF) == 0:
            indicesAF = [i for i in range(amtOptions)]

        energyConsumptionsAF = [cycle['adaptationOptions']['energyConsumption'][i] for i in indicesAF]

        for i in indicesAF:
            if cycle['adaptationOptions']['energyConsumption'][i] == min(energyConsumptionsAF):
                bestOptionAF['energyConsumption'] = cycle['adaptationOptions']['energyConsumption'][i]
                bestOptionAF['latency'] = cycle['adaptationOptions']['latency'][i]
                bestOptionAF['packetLoss'] = cycle['adaptationOptions']['packetLoss'][i]
                break

        # Classification
        indicesTraining = [i for i in range(amtOptions) if cycle['adaptationOptions']['verifiedConfigurations'][i] == 1]

        # TODO move to separate function (same as above)
        indicesClass = [i for i in indicesTraining if cycle['adaptationOptions']['packetLoss'][i] < 10 and cycle['adaptationOptions']['latency'][i] < 5]
        if len(indicesClass) == 0:
            indicesClass = indicesTraining
        energyConsumptionsClass = [cycle['adaptationOptions']['energyConsumption'][i] for i in indicesClass]
        for i in indicesClass:
            if cycle['adaptationOptions']['energyConsumption'][i] == min(energyConsumptionsClass):
                bestOptionClass['energyConsumption'] = cycle['adaptationOptions']['energyConsumption'][i]
                bestOptionClass['latency'] = cycle['adaptationOptions']['latency'][i]
                bestOptionClass['packetLoss'] = cycle['adaptationOptions']['packetLoss'][i]
                break


        amt = len([i for i in range(amtOptions) if cycle['adaptationOptions']['verifiedConfigurations'][i] == 1])
        overallData['amountCL'].append(amt)

        bestConfigurationsACTIVForms.append(bestOptionAF)
        bestConfigurationsClassification.append(bestOptionClass)



    # Written ugly for now
    # TODO fix later
    index = 0
    for cycle in dataRegr:
        bestOptionRegr = {'energyConsumption': max(cycle['adaptationOptions']['energyConsumption']), 'latency': -10, 'packetLoss': -10}

        amtOptions = len(cycle['adaptationOptions']['classificationBefore'])
        

        indicesTraining = [i for i in range(amtOptions) if cycle['adaptationOptions']['verifiedConfigurations'][i] == 1]

        # TODO move to separate function (same as above)
        indicesRegr = [i for i in indicesTraining if cycle['adaptationOptions']['packetLoss'][i] < 10 and cycle['adaptationOptions']['latency'][i] < 5]
        if len(indicesRegr) == 0:
            indicesRegr = indicesTraining
        energyConsumptionsRegr = [cycle['adaptationOptions']['energyConsumption'][i] for i in indicesRegr]
        for i in indicesRegr:
            if cycle['adaptationOptions']['energyConsumption'][i] == min(energyConsumptionsRegr):
                bestOptionRegr['energyConsumption'] = cycle['adaptationOptions']['energyConsumption'][i]
                bestOptionRegr['latency'] = cycle['adaptationOptions']['latency'][i]
                bestOptionRegr['packetLoss'] = cycle['adaptationOptions']['packetLoss'][i]
                break


        amt = len([i for i in range(amtOptions) if cycle['adaptationOptions']['verifiedConfigurations'][i] == 1])
        overallData['amountRE'].append(amt)

        bestConfigurationsRegression.append(bestOptionRegr)
        index += 1


    verificationTimesAF = [cycle['verificationTimeWithoutLearning'] / 1000 for cycle in data]
    verificationTimesCL = [cycle['verificationTimeWithLearning'] / 1000 for cycle in data]
    learningTimesMillisCL = [cycle['learningTime'] for cycle in data]

    verificationTimesRE = [cycle['verificationTimeWithLearning'] / 1000 for cycle in dataRegr]
    learningTimesMillisRE = [cycle['learningTime'] for cycle in dataRegr]

    totalTimesCL = [x+(y/1000) for x,y in zip(verificationTimesCL, learningTimesMillisCL)]
    totalTimesRE = [x+(y/1000) for x,y in zip(verificationTimesRE, learningTimesMillisRE)]

    mlp.rcParams['font.size'] = 14
    plt.figure(figsize=(13,10))
    plt.subplots_adjust(wspace=0.25)

    def makeBoxplot(dataAF, dataCL, dataRE, ylabel):
        boxplot = plt.boxplot(
            [dataAF, dataCL, dataRE],
            labels=['No learning', 'Classification', 'Regression'],
            patch_artist=True,
            widths=0.5,
            medianprops={'color': 'black', 'linewidth': 2}
        )
        for index, box in enumerate(boxplot['boxes']):
            box.set(facecolor=['orange', 'dodgerblue', 'green'][index])
        plt.ylabel(ylabel)
    
    plt.subplot(2,2,1)
    makeBoxplot(
        [i['packetLoss'] for i in bestConfigurationsACTIVForms], 
        [i['packetLoss'] for i in bestConfigurationsClassification], 
        [i['packetLoss'] for i in bestConfigurationsRegression],
        'Packet loss (%)'
    )

    plt.subplot(2,2,2)
    makeBoxplot(
        [i['latency'] for i in bestConfigurationsACTIVForms], 
        [i['latency'] for i in bestConfigurationsClassification], 
        [i['latency'] for i in bestConfigurationsRegression],
        'Latency (%)'
    )

    plt.subplot(2,2,3)
    makeBoxplot(
        [i['energyConsumption'] for i in bestConfigurationsACTIVForms], 
        [i['energyConsumption'] for i in bestConfigurationsClassification], 
        [i['energyConsumption'] for i in bestConfigurationsRegression],
        'Energy consumption (Coulomb)'
    )

    plt.subplot(2,2,4)
    makeBoxplot(
        verificationTimesAF,
        totalTimesCL,
        totalTimesRE,
        'Time (seconds)'
    )

    plt.savefig('OverallEvaluation.pdf', bbox_inches='tight')
    plt.close()


    # reductionPercentagesCL = [overallData['amountCL'][i] / overallData['totalAmount'] for i in range(len(overallData['amountCL']))]
    # reductionPercentagesRE = [overallData['amountRE'][i] / overallData['totalAmount'] for i in range(len(overallData['amountRE']))]
    # verificationPercentagesCL = [totalTimesCL[i]/verificationTimesAF[i] for i in range(len(verificationTimesAF))]
    # verificationPercentagesRE = [totalTimesRE[i]/verificationTimesAF[i] for i in range(len(verificationTimesAF))]

    # print(f'Classification (reduction space): mean = {np.mean(reductionPercentagesCL)*100:.2f} ; std = {np.std(reductionPercentagesCL)*100:.2f}')
    # print(f'Regression (reduction space): mean = {np.mean(reductionPercentagesRE)*100:.2f} ; std = {np.std(reductionPercentagesRE)*100:.2f}')
    # print(f'Classification (reduction verification time): mean = {np.mean(verificationPercentagesCL)*100:.2f} ; std = {np.std(verificationPercentagesCL)*100:.2f}')
    # print(f'Classification (reduction verification time): mean = {np.mean(verificationPercentagesRE)*100:.2f} ; std = {np.std(verificationPercentagesRE)*100:.2f}')

    # print(f'Mean time no learning: {np.mean(verificationTimesAF)}')
    # print(f'Mean time classifier: {np.mean(totalTimesCL)}')
    # print(f'Mean time regressors: {np.mean(totalTimesRE)}')

    # verificationTimeCL = sum(verificationTimesCL)
    # learningTimeCL = sum(learningTimesMillisCL) / 1000
    # print(f"Verification time classifier: {verificationTimeCL:.2f} seconds")
    # print(f"Learning time classifier: {learningTimeCL:.2F} seconds")
    # print(f'Percentage of time spent learning: {learningTimeCL / (learningTimeCL + verificationTimeCL) * 100:.5f} %')

    # Compile data in csv format
    with open('data_overall_results.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['AF_PL', 'AF_LA', 'AF_EC', 'AF_T', 'CL_PL', 'CL_LA', 'CL_EC', 'CL_T', 'RE_PL', 'RE_LA', 'RE_EC', 'RE_T'])

        for i in range(len(bestConfigurationsACTIVForms)):
            writer.writerow([
                bestConfigurationsACTIVForms[i]['packetLoss'],
                bestConfigurationsACTIVForms[i]['latency'],
                bestConfigurationsACTIVForms[i]['energyConsumption'],
                verificationTimesAF[i],

                bestConfigurationsClassification[i]['packetLoss'],
                bestConfigurationsClassification[i]['latency'],
                bestConfigurationsClassification[i]['energyConsumption'],
                totalTimesCL[i],

                bestConfigurationsRegression[i]['packetLoss'],
                bestConfigurationsRegression[i]['latency'],
                bestConfigurationsRegression[i]['energyConsumption'],
                totalTimesRE[i]
            ])



def uncertainties():
    traffic_m10 = [
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, 0.00, 0.07, 0.27, 0.00, 0.00, 0.27, 0.00,
        0.00, 0.07, 0.07, 0.47, 0.00, 0.00, 0.20, 0.07, 0.27, 0.40, 0.00, 0.00, 0.20, 0.20, 0.40, 0.07,
        0.00, 0.20, 0.07, 0.20, 0.00, 0.13, 0.20, 0.20, 0.13, 0.20, 0.13, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ]

    traffic_m13 = [
        0.00, 0.27, 0.27, 0.00, 0.27, 0.00, 0.47, 0.73, 0.40, 0.60, 0.00, 0.00, 0.40, 0.33, 0.00, 0.00,
        0.73, 1.00, 0.73, 0.47, 0.00, 0.13, 0.00, 0.00, 0.47, 1.00, 0.40, 0.73, 0.73, 0.00, 0.60, 0.73,
        0.13, 0.00, 0.00, 0.00, 0.33, 1.00, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ]

    snr_m10_to_m6 = [
        -38.5748536, 2.432525555, 3.039074854, 4.046325223, -
        7.496249251, -2.879919755, -15.42475029,
        -18.60666432, 9.735639693, -14.60678911, -
        14.84345602, -16.23425238, -3.998477817, -8.641358955,
        -20.48427156, -26.78711905, 2.868089362, -
        17.56518085, -2.748258545, -15.74691595, -5.763569926,
        -14.38575714, -16.20248087, -10.44177805, -
        10.99740819, -15.38507103, -5.003991981, -6.498524476,
        -9.755475136, -12.16113134, -9.843062783, 9.769775084, -
        8.111154749, -17.6255459, -0.765304975,
        3.117361662, -14.91365257, -4.995636297, -
        21.46025866, -3.815738489, -10.45276911, -8.102035428,
        -9.385219183, -5.602277673, -11.17232541, -
        3.934254023, 9.19256636, -2.962522429, -3.323884766,
        -2.73437665, -5.165138457, 3.735105121, -
        3.837679463, 9.062083522, -21.14297868, -17.94601833,
        -30.30817648, -24.98122098, -10.93650492, -
        5.483935482, -13.22397488, -5.256657941, 14.42393409,
        -16.9850675, -18.54951675, -12.23788052, -
        11.23186405, -6.07241966, 5.542874266, -12.40390642,
        -7.915286683, -3.326957666, -2.759322742, -
        5.442895106, 1.735612226, -18.66836894, -14.35648924,
        -20.06528816, -12.401528, -13.18512363, -
        18.83655161, -4.936772384, 1.808320838, -14.84545133,
        -19.90588995, 1.640486084, 5.08870118, -
        3.305257736, -10.60481796, 2.208505001, -30.6510474,
        -5.943576113, -19.99982916, -3.012227759, 2.058642615, -4.522109461,
    ]

    snr_m12_to_m3 = [
        -42.19269974, -5.26360474, 5.158714555, 6.133005626, 7.107296698, 6.133005626, 6.372446885,
        6.53758307, 6.347678527, 7.372446885, -6.173423872, -
        8.111154749, -10.45276911, -17.94601833,
        -18.66836894, -10.61944087, -24.17232541, -
        28.14297868, -17.94601833, -30.30817648, -24.98122098,
        -2.970605052, 1.843109805, 5.62565144, 6.101635803, 9.499354779, 9.594445231, 4.592215655, -2.351464625,
        -8.519800047, 2.271158347, 4.695536219, -
        6.117340992, 9.36970719, -3.388427077, -0.130000517, 2.0857247,
        2.621295493, 8.789333603, 1.055665473, -9.249732459, -
        8.38121661, 3.098076577, -14.88519524, -2.956022277,
        -0.871997239, -12.2452035, 8.267650201, -5.244103025, -
        6.271475095, 1.482834337, -1.522581228, -0.917692567,
        16.41852803, -3.285382099, -
        0.220237883, 1.088049478, 12.22706772, 4.259390969, 7.31980924, 8.110193619,
        -3.376939794, 15.68929308, 13.27159803, 3.891324057, -
        4.404266987, 1.410906984, -1.758781049, -0.043501958,
        0.636702421, 8.05418535, 7.975624689, -6.787451083, -
        5.034998753, 0.193673409, -4.928540148, -2.635815017,
        -0.927474598, -1.895278337, -2.616574406, -
        0.514135847, 0.160330695, -7.164146629, -4.32428222, -2.644065586,
        2.112673172, 0.952928027, -16.87634657, -8.255160173, -
        2.382911941, 9.398557263, -0.983030474, 2.517246923,
        -3.305618507, 5.717977803, -7.156165422
    ]

    plt.figure()
    for index, item in enumerate([traffic_m10, traffic_m13, snr_m10_to_m6, snr_m12_to_m3]):
        plt.subplot(2, 2, index + 1)
        plt.plot(np.arange(0, len(item[:76]), 1), item[:76])
        plt.xticks(np.arange(0, len(item[:76]), 25), [
                   '08:00:00', '12:00:00', '16:00:00', '20:00:00'])
        plt.xlabel('Time')
        plt.ylabel(
            ['Traffic Fluctuation', 'Traffic Fluctuation', 'SNR', 'SNR'][index])
        plt.title(['Mote 10', 'Mote 13', 'Mote 10 to Mote 6',
                   'Mote 12 to Mote 3'][index])
        plt.grid()
    plt.show()



# uncertainties()
# overview_of_selected_adaptation_options()
# comparison()
# time_measurement()
# selected_adaptation_options()

# selected_adaptation_options_multigoal(highlightBest=False)
boxplotsResults(ignoreTraining=False)

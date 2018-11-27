# Dependencies: Plotly, psutils, plotly-orca

import sys
import os
from dataLoader import loadData
import csv
from plotly import __version__, tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
import plotly.graph_objs as go
from collections import OrderedDict
from math import log, ceil



CSV_GENERAL_NAME = 'classificationComparison.csv'
CSV_CONFALL_NAME = 'confusionMatricesAll.csv'
CSV_CONFVERS_NAME = 'confusionMatricesVersatile.csv'
PNG_OUTPUT_NAME = 'resultClassificationComparison.png'
HTML_OUTPUT_NAME = 'resultClassificationComparison.html'


def loadDataFromFiles(path):
    data = {}
    files = os.listdir(path)
    for item in files:
        nameClassifier = os.path.splitext(item)[0]
        data[nameClassifier] = loadData(os.path.join(path, item))

    return data


def getErrorRate(configs):
    errorPredictions = [i.getAmtOfWrongPredictions()[0] for i in configs]
    # Use the first configuration to determine the amount of cycles
    amtCycles = len(configs[0])
    return sum(errorPredictions) / (len(errorPredictions) * amtCycles) * 100.0

def getErrorRatePerSide(configs):
    errorRatesPerSides = [i.getWrongPredictionsPerSide()[0] for i in configs]
    errBefore, errAfter = tuple([sum(x) for x in zip(*errorRatesPerSides)])
    totalSamplesPerSide = [i.getAmtResultsPerSide() for i in configs]
    totalBefore, totalAfter = tuple([sum(x) for x in zip(*totalSamplesPerSide)])
    return ((errBefore / totalBefore)*100, (errAfter / totalAfter)*100)


def calculateLogarithmicLoss(configs):
    '''
    Metric used to evaluate machine learning techniques (closer to 0 eqauls better).
    TODO verify if this is usable for our case (binary classification)
    '''
    totalSamples = sum([len(i) for i in configs])
    chancePositiveRes = sum([i.getAmtResultsPerSide()[0] for i in configs]) / totalSamples
    # print(chancePositiveRes)
    logValsConf = lambda x: [log(chancePositiveRes) if i.pl < 10 else log(1-chancePositiveRes) for i in x]
    return (-1 / totalSamples) * sum([sum(logValsConf(config)) for config in configs])


def getCumulativeConfMatrices(configs):
    # TODO might tidy this up later (centralise, also used in dataLoader)
    res = {'TruePositives': 0, 'TrueNegatives': 0, 'FalsePositives': 0, 'FalseNegatives': 0}
    for config in configs:
        confM = config.getConfusionMatrix(technique='classification')
        for key in res.keys():
            res[key] += confM[key]
    return res


def calculateF1Score(confMatrix):
    precision = confMatrix['TruePositives'] / (confMatrix['TruePositives'] + confMatrix['FalsePositives'])
    recall = confMatrix['TruePositives'] / (confMatrix['TruePositives'] + confMatrix['FalseNegatives'])

    return 2 / ((1/precision) + (1/recall))




def printTable(data, outputPath):
    headerColor, rowOddColor, rowEvenColor = '#CCE5FF', '#CCF3FF', 'white'

    cellsData = {'values' : [], 'align': ['left', 'center'], 'height': 30}
    for i in range(len(data['header'])):
        cellsData['values'].append([])

    for i in range(len(data['values'])):
        for j in range(len(data['header'])):
            cellsData['values'][j].append(data['values'][i][j])

    colors = [headerColor] + [rowOddColor if i % 2 else rowEvenColor for i in range(len(data['header'])-1)]
    cellsData['fill'] = {'color': colors}

    trace = go.Table(header={'values' : data['header'], 'fill': {'color':headerColor}}, cells=cellsData, \
        columnwidth=[250, 180])

    layout = dict(width=1800, height=1000, font=dict(family='"Open Sans", verdana, arial, sans-serif', size=18, color='#444'))
    fig = dict(data=[trace], layout=layout)
    plot(fig, filename=os.path.join(outputPath, HTML_OUTPUT_NAME))
    pio.write_image(fig, os.path.join(outputPath, PNG_OUTPUT_NAME))



def printConfusionMatrix(confMatrix, outputPath, filename, technique, loss, scaler):
    # Temporarily used until print of all matrices is fixed
    amtSamples = sum(confMatrix.values())

    trace = go.Table(header={'values': [f'Samples = {amtSamples}', 'Predicted good', 'Predicted bad']}, \
        cells={'values': [['Actually good', 'Actually bad'], \
            [confMatrix['TruePositives'], confMatrix['FalsePositives']], \
            [confMatrix['FalseNegatives'], confMatrix['TrueNegatives']]], \
            'height': 40})
    layout = dict(width=700, height=400, font=dict(family='"Open Sans", verdana, arial, sans-serif', size=18, color='#444'), \
        title=f'Confusion matrix {technique}<br>Scaler={scaler}' + (f', Loss={loss}' if loss != None else ''))
    fig = dict(data=[trace], layout=layout)
    pio.write_image(fig, os.path.join(outputPath, filename + '.png'))


def printConfusionMatrices(confMatrices, outputPath, filename):
    # TODO find out why this will not print more than one table
    s_row = ceil(len(confMatrices)/2)
    # fig = tools.make_subplots(rows=2, cols=s_row, subplot_titles=list(confMatrices.keys()))
    data = []

    for _, confMatrix in sorted(confMatrices.items()):
        index = confMatrices.index(confMatrix)
        amtSamples = sum(confMatrix.values())

        row, col = 1 if index < s_row else 2, (index % s_row) + 1
        trace = go.Table(header={'values': [f'Samples = {amtSamples}', 'Predicted good', 'Predicted bad']}, \
            cells={'values': [['Actually good', 'Actually bad'], \
                [confMatrix['TruePositives'], confMatrix['FalsePositives']], \
                [confMatrix['FalseNegatives'], confMatrix['TrueNegatives']]], \
                'height': 40})
        data.append(trace)
        # fig.append_trace(trace, row, col)

    layout = dict(width=1500, height=1000, font=dict(family='"Open Sans", verdana, arial, sans-serif', size=18, color='#444'), \
        title='Confusion matrices')
    # fig = dict(data=data, layout=layout)
    fig = go.Figure(data=data, layout=layout)
    # fig['layout'].update(width=1500, height=1000, font=dict(family='"Open Sans", verdana, arial, sans-serif', size=18, color='#444'), \
        # title='Confusion matrices')
    # print(data)
    # plot(fig, filename=os.path.join(outputPath, 'tmp.html'))
    pio.write_image(fig, os.path.join(outputPath, filename + '.png'))


def writeConfMatricesToFiles(matrices, filename, outputPath):
    csvOutputFile = open(os.path.join(outputPath, filename), mode='w')
    csvOutputWriter = csv.writer(csvOutputFile, delimiter=',')
    keys = list(list(matrices.values())[0].keys())
    csvOutputWriter.writerow(['technique', 'loss function', 'scaler'] + keys)

    for technique, matrix in sorted(matrices.items()):
        technique, loss, scaler = technique.split('_')
        # Make sure the order of the initial line is kept when storing the data in the csv
        csvOutputWriter.writerow([technique, '-' if loss=='None' else loss, scaler] + [matrix[key] for key in keys])

    csvOutputFile.close()







def compareResultsClassifiers(inputPath, outputPath):
    data = loadDataFromFiles(inputPath)
    header = ['Technique', 'Loss function', 'Scaler','Overall error percentage', 'Error rate < 10% packet loss', 'Error rate > 10% packet loss', \
        'Error rate versatile configurations', 'Error rate < 10% packet loss (versatile configurations)', \
        'Error rate > 10% packet loss (versatile configurations)', 'F1 score (all)', 'F1 score (versatile)']
    outputData = {'header' : header[:4] + header[6:], 'values' : []}

    csvOutputFile = open(os.path.join(outputPath, CSV_GENERAL_NAME), mode='w')
    csvOutputWriter = csv.writer(csvOutputFile, delimiter=',')
    csvOutputWriter.writerow(header)

    confMatrices = {'all': {}, 'versatile': {}}

    # The key is the used classifier, value is the data associated with the classifier
    for key, configurations in sorted(data.items()):
        key = key.split('_')

        if len(key) == 2:
            # The loss function is not applicable in this case
            loss = None
            classifier, scaler = key
        else:
            classifier, loss, scaler = key

        amtCycles = len(configurations[0])

        # The overall percentage of errors in the predicted values
        errorPercentageOverall = getErrorRate(configurations)

        # Explore the error rates for either side of the cutoff line
        errRateBefore, errRateAfter = getErrorRatePerSide(configurations)


        # Explore results in versatile configurations (configurations with less than 80% of results at one side of the line)
        versatileThreshhold = (amtCycles * 0.8) - (amtCycles * 0.2)
        versatileConfigurations = list(filter(lambda x: x.getScatterRate() <= versatileThreshhold, configurations))

        # The error percentage for the versatile configurations
        errorPercentageVersatile = getErrorRate(versatileConfigurations)

        # The error percentage per side for versatile configurations
        errRateVBefore, errRateVAfter = getErrorRatePerSide(versatileConfigurations)


        # The confusion matrices
        # TODO merge all confusion matrices into single plot
        overallConfMatrix, versConfMatrix = getCumulativeConfMatrices(configurations), getCumulativeConfMatrices(versatileConfigurations)
        confMatrices['all'][f'{classifier}_{loss}_{scaler}'] = (overallConfMatrix)
        confMatrices['versatile'][f'{classifier}_{loss}_{scaler}'] = (versConfMatrix)
        printConfusionMatrix(overallConfMatrix, outputPath, f'ConfusionMatrixAll_{classifier}_{loss}_{scaler}', classifier + ' (all)', loss, scaler)
        printConfusionMatrix(versConfMatrix, outputPath, f'ConfusionMatrixVers_{classifier}_{loss}_{scaler}', classifier + ' (versatile)', loss, scaler)

        # F1 values of the configuration (sub)sets
        F1All, F1Vers = calculateF1Score(overallConfMatrix), calculateF1Score(versConfMatrix)

        row = [classifier, '-' if loss==None else loss, scaler, f'{errorPercentageOverall:.2f}%', f'{errRateBefore:.2f}%', f'{errRateAfter:.2f}%', \
            f'{errorPercentageVersatile:.2f}%', f'{errRateVBefore:.2f}%', f'{errRateVAfter:.2f}%', f'{F1All:.4f}', f'{F1Vers:.4f}']
        outputData['values'].append(row[:4] + row[6:])
        csvOutputWriter.writerow(row)
    
    csvOutputFile.close()
    # printConfusionMatrices(confMatrices['all'], outputPath, 'ConfMatricesAll')
    # printConfusionMatrices(confMatrices['versatile'], outputPath, 'ConfMatricesVersatile')

    printTable(outputData, outputPath)
    writeConfMatricesToFiles(confMatrices['all'], CSV_CONFALL_NAME, outputPath)
    writeConfMatricesToFiles(confMatrices['versatile'], CSV_CONFVERS_NAME, outputPath)




if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('This program should be run with 2 commandline arguments:')
        print('\t- The directory where the data of all the classifiers is stored')
        print('\t- The output directory where you want all the files to be saved')
        sys.exit(1)
    
    if not os.path.isdir(sys.argv[2]):
        os.mkdir(sys.argv[2])

    compareResultsClassifiers(sys.argv[1], sys.argv[2])


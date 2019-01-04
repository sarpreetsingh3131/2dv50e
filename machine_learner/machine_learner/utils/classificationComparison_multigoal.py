# Dependencies: Plotly, psutils, plotly-orca

import sys
import os
from dataLoader import loadData
import csv
from sklearn.metrics import f1_score, log_loss, matthews_corrcoef
from plotly import __version__, tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
import plotly.graph_objs as go
from collections import OrderedDict
from math import log, ceil
from printUtils import printProgressBar


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


def getActualAndPredictedClasses(configs):
    trues = []
    predicted = []

    for config in configs:
        for cycle in config:
            actualClass = (1 if cycle.pl < 10 else 0) + 2 * (1 if cycle.la < 5 else 0)
            trues.append(actualClass)
            predicted.append(cycle.clB)

    return (trues, predicted)


def getErrorRate(configs):
    errorPredictions = [i.getAmtOfWrongPredictions()[0] for i in configs]
    # Use the first configuration to determine the amount of cycles
    amtCycles = len(configs[0])
    return sum(errorPredictions) / (len(errorPredictions) * amtCycles) * 100.0


def calculateLogarithmicLoss(configs):
    '''
    Metric used to evaluate machine learning techniques (closer to 0 eqauls better).
    '''
    return None



def getConfusionMatrix(configs):
    # The first index indicates the real class, the second index indicates the predicted class
    confMatrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    
    for config in configs:
        for cycle in config:
            actualClass = (1 if cycle.pl < 10 else 0) + 2 * (1 if cycle.la < 5 else 0)
            predictedClass = cycle.clB
            confMatrix[actualClass][predictedClass] += 1
    
    return confMatrix


def calculateF1Scores(configs):
    actual, predicted = getActualAndPredictedClasses(configs)
    return f1_score(actual, predicted, average=None)

def calculateF1Score(configs):
    actual, predicted = getActualAndPredictedClasses(configs)
    return f1_score(actual, predicted, average='weighted')


def calculateMatthewsCorrCoef(configs):
    actual, predicted = getActualAndPredictedClasses(configs)
    return matthews_corrcoef(actual, predicted)


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
        columnwidth=[270, 240, 220, 150, 150, 150, 300])

    layout = dict(width=1800, height=1000, font=dict(family='"Open Sans", verdana, arial, sans-serif', size=18, color='#444'))
    fig = dict(data=[trace], layout=layout)
    pio.write_image(fig, os.path.join(outputPath, PNG_OUTPUT_NAME))
    plot(fig, filename=os.path.join(outputPath, HTML_OUTPUT_NAME))



def printConfusionMatrix(confMatrix, outputPath, filename, technique, loss, scaler):
    # Temporarily used until print of all matrices is fixed
    sumNested = lambda x: sum(map(sumNested, x)) if isinstance(x, list) else x
    amtSamples = sumNested(confMatrix)
    rows = {'values' : [[],[],[],[],[]], 'height':40}
    rows['values'][0] = ['Actually class 0','Actually class 1','Actually class 2','Actually class 3']
    for i in range(len(confMatrix)):
        for j in range(len(confMatrix[i])):
            rows['values'][j+1].append(confMatrix[i][j])

    trace = go.Table(header={'values': [f'Samples = {amtSamples}', 'Predicted class 0', 'Predicted class 1', 'Predicted class 2', 'Predicted class 3']}, cells=rows)

    layout = dict(width=700, height=600, font=dict(family='"Open Sans", verdana, arial, sans-serif', size=18, color='#444'), \
        title=f'Confusion matrix {technique}<br>Scaler={scaler}' + (f', Loss={loss}' if loss != None else ''))
    fig = dict(data=[trace], layout=layout)
    pio.write_image(fig, os.path.join(outputPath, filename + '.png'))
    


def writeConfMatricesToFiles(matrices, filename, outputPath):
    if len(matrices.items()) == 0:
        return

    csvOutputFile = open(os.path.join(outputPath, filename), mode='w')
    csvOutputWriter = csv.writer(csvOutputFile, delimiter=',')
    keys = []
    for i in range(4):
        for j in range(4):
            keys.append(f'Actual{i} - Predicted{j}')

    csvOutputWriter.writerow(['technique', 'loss function', 'scaler'] + keys)

    for technique, matrix in sorted(matrices.items()):
        technique, loss, scaler = technique.split('_')
        # Make sure the order of the initial line is kept when storing the data in the csv
        confValues = []
        for i in range(4):
            for j in range(4):
                confValues.append(matrix[i][j])
        csvOutputWriter.writerow([technique, '-' if loss=='None' else loss, scaler] + confValues)

    csvOutputFile.close()







def compareResultsClassifiers(inputPath, outputPath):
    files = sorted(os.listdir(inputPath))
    bestSample = (100, '')

    header = [
        'Technique',
        'Loss function',
        'Scaler',
        'Overall error percentage',
        'F1 score (weighted)',
        'Matthews correlation coefficient',
        'F1 scores (class 0 - 3)'
    ]
    
    outputData = {'header' : header, 'values' : []}

    csvOutputFile = open(os.path.join(outputPath, CSV_GENERAL_NAME), mode='w')
    csvOutputWriter = csv.writer(csvOutputFile, delimiter=',')
    csvOutputWriter.writerow(header)

    confMatrices = {'all': {}, 'versatile': {}}

    index = 0
    printProgressBar(index, len(files), prefix='Processing of classifiers:', suffix='Complete', length=30)
    # The key is the used classifier, value is the data associated with the classifier
    for filename in files:
        configurations = loadData(os.path.join(inputPath, filename))
        key = os.path.splitext(filename)[0]
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

        # Skip the classifiers with an error rate over 50%
        if errorPercentageOverall > 30:
            # NOTE: removes the file (make sure it is stored somewhere else as well)
            os.remove(os.path.join(inputPath,filename))
            index += 1
            printProgressBar(index, len(files), prefix='Processing of classifiers:', suffix='Complete', length=30)
            continue
            
        if errorPercentageOverall < bestSample[0]:
            bestSample = (errorPercentageOverall, filename)


        # Confusion matrix for all configurations
        overallConfMatrix = getConfusionMatrix(configurations)
        confMatrices['all'][f'{classifier}_{loss}_{scaler}'] = (overallConfMatrix)
        printConfusionMatrix(overallConfMatrix, outputPath, f'ConfusionMatrixAll_{classifier}_{loss}_{scaler}', classifier, loss, scaler)
        

        matthewsCorrCoef = calculateMatthewsCorrCoef(configurations)

        # F1 values of all the configurations
        F1All = calculateF1Scores(configurations)
        F1AllWeighted = calculateF1Score(configurations)
        F1AllStr = '[' + ','.join([f'{i:.4f}' for i in F1All]) + ']'
        del configurations

        row = [
            classifier,
            '-' if loss == None else loss,
            scaler,
            f'{errorPercentageOverall:.2f}%',
            f'{F1AllWeighted:.4f}',
            f'{matthewsCorrCoef:.4f}',
            f'{F1AllStr}'
        ]
        outputData['values'].append(row)
        csvOutputWriter.writerow(row)

        index += 1
        printProgressBar(index, len(files), prefix='Processing of classifiers:', suffix='Complete', length=30)

    
    csvOutputFile.close()

    print()
    print(f'Best sample ({bestSample[0]:.2f}%): {bestSample[1]}')
    print(f'Best Matthews correlation coefficient: {max([float(i[5]) for i in outputData["values"]]):.4f}')
    print(f'Best F1 score (weighted): {max([float(i[4]) for i in outputData["values"]]):.4f}')

    writeConfMatricesToFiles(confMatrices['all'], CSV_CONFALL_NAME, outputPath)
    printTable(outputData, outputPath)




if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('This program should be run with 2 commandline arguments:')
        print('\t- The directory where the data of all the classifiers is stored')
        print('\t- The output directory where you want all the files to be saved')
        sys.exit(1)
    
    if not os.path.isdir(sys.argv[2]):
        os.mkdir(sys.argv[2])

    compareResultsClassifiers(sys.argv[1], sys.argv[2])


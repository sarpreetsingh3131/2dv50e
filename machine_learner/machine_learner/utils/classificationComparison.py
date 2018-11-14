# Dependencies: Plotly, psutils, plotly-orca

import sys
import os
from dataLoader import loadData
import csv
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
import plotly.graph_objs as go
from collections import OrderedDict



CSV_OUTPUT_NAME = 'classificationComparison.csv'
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



def compareResultsClassifiers(inputPath, outputPath):
    data = loadDataFromFiles(inputPath)
    header = ['Technique','Overall error percentage', 'Error rate < 10% packet loss', 'Error rate > 10% packet loss', \
        'Error rate versatile configurations', 'Error rate < 10% packet loss (versatile configurations)', \
        'Error rate > 10% packet loss (versatile configurations)']
    outputData = {'header' : header[:2] + header[4:], 'values' : []}

    csvOutputFile = open(os.path.join(outputPath, CSV_OUTPUT_NAME), mode='w')
    csvOutputWriter = csv.writer(csvOutputFile, delimiter=',')
    csvOutputWriter.writerow(header)

    # The key is the used classifier, value is the data associated with the classifier
    for key, configurations in sorted(data.items()):
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


        # print(f'Results for classification technique: {key}')
        # print(f'\tOverall error rate classification: {errorPercentageOverall:.2f}%')
        # print(f'\tError rates before/after cutoff line respectively: {errRateBefore:.2f}% - {errRateAfter:.2f}%')
        # print(f'\tError rate for versatile configurations: {errorPercentageVersatile:.2f}%')
        # print(f'\tError rates before/after cutoff line respectively: {errRateVBefore:.2f}% - {errRateVAfter:.2f}%')


        row = [key, f'{errorPercentageOverall:.2f}%', f'{errRateBefore:.2f}%', f'{errRateAfter:.2f}%', f'{errorPercentageVersatile:.2f}%', \
            f'{errRateVBefore:.2f}%', f'{errRateVAfter:.2f}%']
        outputData['values'].append(row[:2] + row[4:])
        csvOutputWriter.writerow(row)
    
    csvOutputFile.close()
    printTable(outputData, outputPath)


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

    trace = go.Table(header={'values' : data['header'], 'fill': {'color':headerColor}}, cells=cellsData)

    layout = dict(width=1700, height=400, font=dict(family='"Open Sans", verdana, arial, sans-serif', size=18, color='#444'))
    fig = dict(data=[trace], layout=layout)
    plot(fig, filename=os.path.join(outputPath, HTML_OUTPUT_NAME))
    pio.write_image(fig, os.path.join(outputPath, PNG_OUTPUT_NAME))



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('This program should be run with 2 commandline arguments:')
        print('\t- The directory where the data of all the classifiers is stored')
        print('\t- The output directory where you want all the files to be saved')
        sys.exit(1)
    
    if not os.path.isdir(sys.argv[2]):
        os.mkdir(sys.argv[2])

    compareResultsClassifiers(sys.argv[1], sys.argv[2])


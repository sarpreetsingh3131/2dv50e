import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import functools
import sys
import os
import shutil
from math import ceil
from dataLoader import loadData

PLOT_OUTPUT_DIR = ''


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)

    Reference: https://stackoverflow.com/a/34325723
    """
    percent = 100 * (iteration / float(total))
    percentStr = f'{percent:.1f}'
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percentStr, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()




def analyseUncertainties():
    pathFile = os.path.join('machine_learner', 'collected_data', 'overall_adaptation_options.json')

    adapResults = loadData(pathFile)

    
    # Get the minimum and maximum value for energy consumption over all configurations (used in graphs)
    ecResults = [[ao.ec for ao in adapResult] for adapResult in adapResults]
    # flatten the list
    ecResults = [item for sublist in ecResults for item in sublist]
    minEC, maxEC = min(ecResults), max(ecResults)

    worstPredictionRate = sorted([result for result in adapResults], key=lambda res: res.getAmtOfWrongPredictions()[0], reverse=True)
    print()


    # ===================
    # Plotting of results
    # ===================

    # Plots for the 20 adaptation options with the worst predictions
    for i in range(20):
        pass
        # plotLearningEvolution(worstPredictionRate[i], minEC, maxEC, f'worstPrediction{i+1}')
        # printProgressBar(i+1, 20, prefix="Progress worst prediction graphs:", suffix='Complete', length=30)

    # Plots for all adaptation options
    for i in range(len(adapResults)):
        plotLearningEvolution(adapResults[i], minEC, maxEC, f'all{i+1}')
        printProgressBar(i+1, len(adapResults), prefix="Progress all graphs:", suffix='Complete', length=30)



        


def plotLearningEvolution(item, minEC, maxEC, graphName):
    outputDir = createOrGetDir('LearningEvolution')

    dataPacketLoss = [ao.pl for ao in item]
    dataLatency = [ao.la for ao in item]

    minPL, maxPL = min(dataPacketLoss), max(dataPacketLoss)
    minLA, maxLA = min(dataLatency), max(dataLatency)

    # Classification
    fig = initialiseFigure('Effect of online learning adjustments (1 configuration, classification)')
    step = ceil(len(item) / 3)
    for i in range(6):
        cutOffA, cutOffB = (i%3) * step, min(((i%3)+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+1)
        if i == 1:
            plt.legend(
                bbox_to_anchor=(-0.5, 1.1, 2., .122),
                # bbox_to_anchor=(-0.5, 1.08, 2., .122),
                ncol=2,
                mode='expand',
                fontsize='large',
                handles=[
                    mpatches.Patch(color='xkcd:blue', label='Class 3 (packet loss & latency goals)'),
                    mpatches.Patch(color='xkcd:lime green', label='Class 2 (latency goal only)'),
                    mpatches.Patch(color='xkcd:lavender', label='Class 1 (packet loss goal only)'),
                    mpatches.Patch(color='xkcd:orange', label='Class 0 (no goals)')
                ]
            )
        stageLearning = 'before' if i < 3 else 'after'
        ax.set_title(f'Cycles [{cutOffA}:{cutOffB}] {stageLearning} online learning')

        for point in subPoints:
            valToComp = point.clB if i < 3 else point.clA
            color = 'xkcd:orange'
            if valToComp == 3:
                color = 'xkcd:blue'
            elif valToComp == 2:
                color = 'xkcd:lime green'
            elif valToComp == 1:
                color = 'xkcd:lavender' # TODO: find better colour than this

            # TODO: could vary size/color opacity here to indicate how good the energy consumption of this point is
            plt.scatter(point.pl, point.la, color=color, s=20)
            
        # Line for packet loss
        plt.plot([10, 10], [minLA, maxLA], color='red')
        # Line for latency
        plt.plot([minPL, maxPL], [5,5], color='red')

        adjustLabels('Packet Loss (%)', 'Latency (%)')
        adjustXAxis((minPL, maxPL))
        adjustYAxis((minLA, maxLA))

    plt.savefig(os.path.join(outputDir, f'LearningEvolution_classification_{graphName}.png'), bbox_inches='tight')
    plt.close()



def adjustXAxis(target):
    '''
        Helper function to adjust the X axis values of the plot
    '''
    _, _, minY, maxY = plt.axis()
    plt.axis([target[0], target[1], minY, maxY])

def adjustYAxis(target):
    '''
        Helper function to adjust the Y axis values of the plot
    '''
    minX, maxX, _, _ = plt.axis()
    plt.axis([minX, maxX, target[0], target[1]])

def adjustLabels(labelX='Packet Loss (%)', labelY='(Energy Consumption (coulomb)'):
    '''
        Helper function to adjust the plot's axis labels
    '''
    plt.xlabel(labelX)
    plt.ylabel(labelY)


def createOrGetDir(dirName):
    '''
        Helper function that creates a directory relative to PLOT_OUTPUT_DIR if it does not exist yet.
        @returns the path to this (created) directory.
    '''
    outputDir = os.path.join(PLOT_OUTPUT_DIR, dirName)
    if not(os.path.isdir(outputDir)):
        os.mkdir(outputDir)
    return outputDir


def initialiseFigure(title, fontsize=18, res=(1920, 1080), dpi=96):
    '''
        Helper function to initialise the figure with the correct resolution and title.
        @returns the figure.
    '''
    # TODO figure out why the resolution seems to be off (actual resolution is 1568x999)
    fig = plt.figure(figsize=(res[0]/dpi, res[1]/dpi), dpi=dpi)
    fig.suptitle(title, fontsize=fontsize)
    return fig



if __name__ == '__main__':
    # NOTE: this program removes all files from the provided/default folder recursively
    #       -> be careful when providing a custom directory
    
    if len(sys.argv) != 2:
        print("Output directory not provided as commandline argument, using './GraphOutputs' by default")
        PLOT_OUTPUT_DIR = os.path.join('GraphOutputs')
    else:
        PLOT_OUTPUT_DIR = sys.argv[1]

    if os.path.isdir(PLOT_OUTPUT_DIR):
        shutil.rmtree(PLOT_OUTPUT_DIR)

    os.mkdir(PLOT_OUTPUT_DIR)

    analyseUncertainties()

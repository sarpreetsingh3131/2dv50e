import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import functools
import sys
import os
import shutil
from dataLoader import loadData
from printUtils import printProgressBar

PLOT_OUTPUT_DIR = ''



def analyseUncertainties():
    pathFile = os.path.join('machine_learner', 'collected_data', 'overall_adaptation_options.json')

    adapResults = loadData(pathFile)

    
    # Get the minimum and maximum value for energy consumption over all configurations (used in graphs)
    ecResults = [[ao.ec for ao in adapResult] for adapResult in adapResults]
    # flatten the list
    ecResults = [item for sublist in ecResults for item in sublist]
    minEC, maxEC = min(ecResults), max(ecResults)

    # sort the items of interest in ascending order
    # sortFunction = lambda x, y: -1 if x.getScatterRate() < y.getScatterRate() else (0 if x.getScatterRate() == y.getScatterRate() else 1)
    # worstScatterRate = sorted([result for result in adapResults], key=functools.cmp_to_key(sortFunction), reverse=False)


    # sortFunction = lambda x, y: -1 if x.getAmtOfWrongPredictions() < y.getAmtOfWrongPredictions() else (0 if x.getAmtOfWrongPredictions() == y.getAmtOfWrongPredictions() else 1)
    # worstPredictionRate = sorted([result for result in adapResults], key=lambda res: res.getAmtOfWrongPredictions()[0], reverse=True)
    print()


    # ===================
    # Plotting of results
    # ===================

    # Plots for the 5 adaptation options with the worst scatter rate
    for i in range(5):
        pass
        # plotRegressionPredictions(worstScatterRate[i], minEC, maxEC, f'worstScatter{i+1}')
        # plotLearningEvolution(worstScatterRate[i], minEC, maxEC, f'worstScatter{i+1}')
        # plotEffectUncertainties(worstScatterRate[i], f'worstScatter{i+1}')
        # plotLearningEffectOneCycle(worstScatterRate[i], minEC, maxEC, f'worstScatter{i+1}')
        # plotLearningEffect2Cycles(worstScatterRate[i], minEC, maxEC, f'worstScatter{i+1}')
        # printProgressBar(i+1, 5, prefix="Progress worst scatter rates:", suffix='Complete', length=30)


    # Plots for the 20 adaptation options with the worst predictions
    for i in range(20):
        pass
        # print(f"Error rate configuration {i+1}: {worstPredictionRate[i].getAmtOfWrongPredictions()}")
        # plotLearningEvolution(worstPredictionRate[i], minEC, maxEC, f'worstPrediction{i+1}')
        # plotEffectUncertainties(worstPredictionRate[i], f'worstPrediction{i+1}')
        # printProgressBar(i+1, 20, prefix="Progress worst prediction graphs:", suffix='Complete', length=30)

    # Plots for adaptationOptions which have at most 100 samples more on either side of the cutoff line in comparison to the other side
    filteredOptions = list(filter(lambda x: x.getScatterRate() <= 100, adapResults))
    for i in range(len(filteredOptions)):
        pass
        # plotLearningEvolution(filteredOptions[i], minEC, maxEC, f'filtered{i+1}')
    
    # Plots for all adaptation options
    for i in range(len(adapResults)):
        plotRegressionPredictions(adapResults[i], minEC, maxEC, f'all{i+1}')
        plotLearningEvolution(adapResults[i], minEC, maxEC, f'all{i+1}')
        printProgressBar(i+1, len(adapResults), prefix="Progress all graphs:", suffix='Complete', length=30)





def plotLearningEffectOneCycle(item, minEC, maxEC, graphName):
    dataPacketLoss = [ao.pl for ao in item]
    dataRegression = [ao.reB for ao in item]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    initialiseFigure('Effect of online learning adjustments', fontsize=20)
    
    step = int((len(item)-1) / 6)
    for i in range(step, len(item), step):
        plt.subplot(2, 3, int(i/step))

        plt.scatter(item[i].reB, item[i].ec, color='red', s=25, label='Before learning')
        plt.scatter(item[i].reA, item[i].ec, color='green', s=25, label='After learning')
        plt.scatter(item[i].pl, item[i].ec, color='orange', s=25, label='ActivForms')
        plt.legend()

        adjustLabels()
        adjustXAxis((minPL, maxPL))


    outputDir = createOrGetDir('LearningEffect1Cycle')
    plt.savefig(os.path.join(outputDir, f'effectLearning1Cycle_{graphName}.png'), bbox_inches='tight')
    plt.close()



def plotLearningEffect2Cycles(item, minEC, maxEC, graphName):
    dataPacketLoss = [ao.pl for ao in item[0:60]]
    dataRegression = [ao.reB for ao in item[0:60]]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    initialiseFigure('Effect of online learning adjustments over 2 cycles', fontsize=20)
    
    # chosen arbitrarily
    # TODO use heuristic to find interesting points
    step = int((len(item)-1) / 6)
    for i in range(step, len(item), step):
        ax = plt.subplot(2, 3, int(i/step))

        # Plot the regression predictions before and after learning for the first cycle
        x1,y1 = item[i-1].reB, item[i-1].ec
        x2,y2 = item[i-1].reA, item[i-1].ec

        plt.scatter(x1, y1, color='xkcd:red', s=50, label='Regression before learning')
        plt.scatter(x2, y2, color='xkcd:green', s=50, label='Regression after learning')
        
        ax.add_patch(mpatches.FancyArrowPatch((x1, y1), (x2, y2), mutation_scale=8))

        # Plot the second cycle similar to the first cycle
        x1,y1 = item[i].reB, item[i].ec
        x2,y2 = item[i].reA, item[i].ec

        plt.scatter(x1, y1, color='xkcd:brick red', s=50)
        plt.scatter(x2, y2, color='xkcd:leaf green', s=50)
        
        ax.add_patch(mpatches.FancyArrowPatch((x1, y1), (x2, y2), mutation_scale=8))


        # Plot the activforms values
        x1,y1 = item[i-1].pl, item[i-1].ec
        x2,y2 = item[i].pl, item[i].ec

        plt.scatter(x1, y1, color='xkcd:orange', s=50, label='ActivFORMS')
        plt.scatter(x2, y2, color='xkcd:dark orange', s=50)
        
        ax.add_patch(mpatches.FancyArrowPatch((x1, y1), (x2, y2), mutation_scale=8))

        plt.legend()

        adjustLabels()
        adjustXAxis((minPL-0.5, maxPL+0.5))

    
    outputDir = createOrGetDir('LearningEffect2Cycles')
    plt.savefig(os.path.join(outputDir, f'LearningEffect2Cycles_{graphName}.png'), bbox_inches='tight')
    plt.close()



def plotEffectUncertainties(item, graphName):

    dataPacketLoss = [ao.pl for ao in item[0:60]]
    dataRegression = [ao.reB for ao in item[0:60]]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    dataEnergyConsumption = [ao.ec for ao in item[0:60]]
    minEC, maxEC = min(dataEnergyConsumption), max(dataEnergyConsumption)

    initialiseFigure('Effect of uncertainties on adaptation option')

    # Plot 10 data points per graph
    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        ax.set_title(f'Cycles [{i*10}:{((i+1)*10)}]')

        # plot the arrows between the points
        for j in range(i * 10, (i * 10) + 9):
            first = item[j]
            second = item[j+1]
            ax.add_patch(mpatches.FancyArrowPatch((first.pl, first.ec), (second.pl, second.ec), mutation_scale=8))

        # plot the points
        for j in range(i * 10, (i + 1) * 10):
            plt.scatter(item[j].pl, item[j].ec, color='green' if j != (i*10) else 'purple' , s=25)

        plt.xlabel('Packet Loss (%)')
        plt.ylabel('Energy Consumption (coulomb)')
        plt.plot([10, 10], [minEC-0.1,maxEC+0.1], color='red')
        plt.axis([minPL-0.5, maxPL+0.5, minEC-0.1, maxEC+0.1])
    

    outputDir = createOrGetDir('EffectUncertainties')
    plt.savefig(os.path.join(outputDir, f'EffectUncertainties_{graphName}.png'), bbox_inches='tight')
    plt.close()


        

def plotRegressionPredictions(item, minEC, maxEC, graphName):
    dataPacketLoss = [ao.pl for ao in item]
    dataRegression = [ao.reB for ao in item]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    initialiseFigure('Regression predictions vs actual values (single configuration)')

    #plot the result which is spread out the most over energy consumption
    plt.subplot(2, 2, 1)
    [plt.scatter(ao.pl, ao.ec, color='green', s=10) for ao in item]
    plt.plot([10, 10], [minEC, maxEC], color='red')
    plt.xlabel('Packet Loss (%)')
    plt.ylabel('Energy Consumption (coulomb)')
    _, _, minY, maxY = plt.axis()
    plt.axis([minPL, maxPL, minY, maxY])
    plt.legend(
        bbox_to_anchor=(0.3, 1.02, 1.6, .102),
        loc=9,
        ncol=2,
        mode='expand',
        borderaxespad=-2.,
        handles=[
            mpatches.Patch(color='green', label='ActivForms'),
            mpatches.Patch(color='xkcd:sky blue', label='Regression')
        ]
    )


    plt.subplot(2, 2, 2)
    [plt.scatter(ao.reB, ao.ec, color='xkcd:sky blue', s=10) for ao in item]
    plt.plot([10, 10], [minEC, maxEC], color='red')
    plt.xlabel('Packet Loss (%)')
    plt.ylabel('Energy Consumption (coulomb)')
    _, _, minY, maxY = plt.axis()
    plt.axis([minPL, maxPL, minY, maxY])


    plt.subplot(2, 2, 3)
    plt.hist(dataPacketLoss, 50, facecolor='green', edgecolor='black', linewidth=1.2)
    _, _, _, maxY = plt.axis()
    plt.axis([minPL, maxPL, 0, maxY])
    plt.xlabel('Packet Loss (%)')
    plt.ylabel('Amount of samples')

    plt.subplot(2, 2, 4)
    plt.hist(dataRegression, 50, facecolor='xkcd:sky blue',  edgecolor='black', linewidth=1.2)
    _, _, _, maxY = plt.axis()
    plt.axis([minPL, maxPL, 0, maxY])
    plt.xlabel('Packet Loss (%)')
    plt.ylabel('Amount of samples')


    outputDir = createOrGetDir('RegressionPredictions')
    plt.savefig(os.path.join(outputDir, f'RegressionPredictions_{graphName}.png'), bbox_inches='tight')
    plt.close()



def plotLearningEvolution(item, minEC, maxEC, graphName):
    outputDir = createOrGetDir('LearningEvolution')

    dataPacketLoss = [ao.pl for ao in item]
    dataRegression = [ao.reB for ao in item]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)


    initialiseFigure('Effect of online learning adjustments (1 configuration, regression)')

    step = int(len(item) / 3)
    for i in range(6):
        cutOffA, cutOffB = (i%3) * step, min(((i%3)+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+1)
        if i == 1:
            plt.legend(
                bbox_to_anchor=(-0.5, 1.08, 2., .122),
                ncol=2,
                mode='expand',
                fontsize='large',
                handles=[
                    mpatches.Patch(color='xkcd:green', label='Positive prediction from regressor'),
                    mpatches.Patch(color='xkcd:orange', label='Negative prediction from regressor')
                ]
            )
        stageLearning = 'before' if i < 3 else 'after'
        ax.set_title(f'Cycles [{cutOffA}:{cutOffB}] {stageLearning} online learning')

        for point in subPoints:
            valToComp = point.reB if i < 3 else point.reA
            color = 'xkcd:orange' if valToComp >= 10 else 'xkcd:green'
            plt.scatter(point.pl, point.ec, color=color, s=20)

        _, _, minY, maxY = plt.axis()
        plt.plot([10, 10], [minY, maxY], color='red')
        plt.xlabel('Packet Loss (%)')
        plt.ylabel('Energy Consumption (coulomb)')
        adjustXAxis((minPL, maxPL))

    plt.savefig(os.path.join(outputDir, f'LearningEvolution_regression_{graphName}.png'), bbox_inches='tight')
    plt.close()


    # Classification part, similar to regression
    fig = initialiseFigure('Effect of online learning adjustments (1 configuration, classification)')
    step = int(len(item) / 3)
    for i in range(6):
        cutOffA, cutOffB = (i%3) * step, min(((i%3)+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+1)
        if i == 1:
            plt.legend(
                bbox_to_anchor=(-0.5, 1.08, 2., .122),
                ncol=2,
                mode='expand',
                fontsize='large',
                handles=[
                    mpatches.Patch(color='xkcd:blue', label='Positive prediction from classifier'),
                    mpatches.Patch(color='xkcd:orange', label='Negative prediction from classifier')
                ]
            )
        stageLearning = 'before' if i < 3 else 'after'
        ax.set_title(f'Cycles [{cutOffA}:{cutOffB}] {stageLearning} online learning')

        for point in subPoints:
            valToComp = point.clB if i < 3 else point.clA
            color = 'xkcd:orange' if valToComp == 0 else 'xkcd:blue'
            plt.scatter(point.pl, point.ec, color=color, s=20)

        _, _, minY, maxY = plt.axis()
        plt.plot([10, 10], [minY, maxY], color='red')
        plt.xlabel('Packet Loss (%)')
        plt.ylabel('Energy Consumption (coulomb)')
        adjustXAxis((minPL, maxPL))

    plt.savefig(os.path.join(outputDir, f'LearningEvolution_classification_{graphName}.png'), bbox_inches='tight')
    plt.close()



def getMinMax(arr1, arr2):
    '''
        Helper function to determine the minimum and maximum value of 2 lists
    '''
    return ( min(min(arr1), min(arr2)), max(max(arr1), max(arr2)) )


def adjustXAxis(target):
    '''
        Helper function to adjust the X axis values of the plot
    '''
    _, _, minY, maxY = plt.axis()
    plt.axis([target[0], target[1], minY, maxY])


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
    if len(sys.argv) != 2:
        print("Output directory not provided as commandline argument, using './GraphOutputs' by default")
        PLOT_OUTPUT_DIR = os.path.join('GraphOutputs')
    else:
        PLOT_OUTPUT_DIR = sys.argv[1]

    # if os.path.isdir(PLOT_OUTPUT_DIR):
    #     shutil.rmtree(PLOT_OUTPUT_DIR)

    # os.mkdir(PLOT_OUTPUT_DIR)

    analyseUncertainties()

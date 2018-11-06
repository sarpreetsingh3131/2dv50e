import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import functools
import sys
import os
import shutil

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
    """
    percent = 100 * (iteration / float(total))
    percentStr = f'{percent:.1f}'
    # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percentStr, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

class AdaptationResult:
    def __init__(self, ec, pl, clB, reB, clA, reA):
        '''
            ec = Energy consumption (as evaluated by uppaal)
            pl = packet loss (as evaluated by uppaal)
            clB = classification prediction before online training adjustment
            clA = classification prediction after online training adjustment
            reB = regression prediction before online training adjustment
            reA = regression prediction after online training adjustment
        '''
        self.ec = ec
        self.pl = pl
        self.clB = clB
        self.reB = reB
        self.clA = clA
        self.reA = reA

class AdaptationResults:
    def __init__(self, adaptationIndex):
        self.results = []
        self.index = adaptationIndex

    def __iter__(self):
        for i in self.results:
            yield i

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            return self.results[indices]
        raise Exception()

    def __len__(self):
        return len(self.results)

    def addResult(self, ec, pl, clB, reB, clA, reA):
        self.results.append(AdaptationResult(ec, pl, clB, reB, clA, reA))

    def getResults(self):
        return self.results

    # Gets a factor used to determine how much the sample is scattered between the cutoff line of packet loss
    # This is computed as follows: abs((#samples <= 10%) - (#samples > 10%))
    # The lower the factor, the more the results of this adaptation option is spread out over this cutoff line
    def getScatterRate(self):
        tmp = [i.pl for i in self.results]
        return abs((sum(1 if i < 10.0 else 0 for i in tmp)*2) - len(tmp))


    # Computes the maximum distance of 2 distinct result
    def getMaxDistPL(self):
        tmp = [i.pl for i in self.results]
        return max(tmp) - min(tmp)

    # Gets the mean distance of all tuples (x1, x2) 
    def getMeanDistPL(self):
        # NOTE: this functions assumes that the amount of cycles run is even
        sortedList = np.array(sorted([i.pl for i in self.results]))
        return np.mean([sortedList[-(i + 1)] - sortedList[i] for i in range(len(sortedList) >> 1)])
        

    def getMaxDistEC(self):
        tmp = [i.ec for i in self.results]
        return max(tmp) - min(tmp)




def analyseUncertainties():
    pathFile = os.path.join('machine_learner', 'collected_data', 'overall_adaptation_options.json')
    data = json.load(open(pathFile))

    # # Load the adaptation options once at startup
    adapResults = []
    for i in range(len(data[0]['adaptationOptions'])):
        adapResults.append(AdaptationResults(i))

    for i in range(len(data)):
        dataCycleI = data[i]['adaptationOptions']

        for adaptationOption in range(len(dataCycleI)):
            ai, pl, ec, clB, reB, clA, reA = \
                dataCycleI[adaptationOption]['adaptationOption'], \
                dataCycleI[adaptationOption]['packetLoss'], \
                dataCycleI[adaptationOption]['energyConsumption'], \
                dataCycleI[adaptationOption]['classificationBefore'], \
                dataCycleI[adaptationOption]['regressionBefore'], \
                dataCycleI[adaptationOption]['classificationAfter'], \
                dataCycleI[adaptationOption]['regressionAfter']

            # FIXME Amount of learning cycles hardcoded for now
            if i > 29:
                adapResults[ai].addResult(ec, pl, clB, reB, clA, reA)


    # plot the result with the lowest spreading factor
    ecResults = [[ao.ec for ao in adapResult] for adapResult in adapResults]
    # flatten the list
    ecResults = [item for sublist in ecResults for item in sublist]
    minEC, maxEC = min(ecResults), max(ecResults)

    # sort the items of interest in descending order
    sortFunction = lambda x, y: -1 if x.getScatterRate() < y.getScatterRate() else (0 if x.getScatterRate() == y.getScatterRate() else 1)
    sortedItems = sorted([result for result in adapResults], key=functools.cmp_to_key(sortFunction), reverse=False)

    for i in range(5):
        # plotRegressionPredictions(sortedItems[i], minEC, maxEC)
        plotLearningEvolution(sortedItems[i], minEC, maxEC, f'worst{i+1}')
        plotEffectUncertainties(sortedItems[i], f'worst{i+1}')
        # plotLearningEffectOneCycle(sortedItems[i], minEC, maxEC, f'worst{i+1}')
        # plotLearningEffect2Cycles(sortedItems[i], minEC, maxEC, f'worst{i+1}')

    # AdapOptions which have at most 100 samples more on either side of the cutoff line in comparison to the other side
    filteredOptions = list(filter(lambda x: x.getScatterRate() <= 100, adapResults))
    for i in range(len(filteredOptions)):
        pass
        # plotLearningEvolution(filteredOptions[i], minEC, maxEC, f'filtered{i+1}')
    
    printProgressBar(0, len(adapResults) - 1, prefix="Progress all graphs:", suffix='Complete', length=50)
    for i in range(len(adapResults)):
        plotLearningEvolution(adapResults[i], minEC, maxEC, f'all{i+1}')
        printProgressBar(i, len(adapResults) - 1, prefix="Progress all graphs:", suffix='Complete', length=50)



def plotLearningEffectOneCycle(item, minEC, maxEC, graphName):
    dataPacketLoss = [ao.pl for ao in item]
    dataRegression = [ao.reB for ao in item]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    # TODO get these values right -> 16:9 ratio (not sure why the ratio is off, maybe rounding of the float numbers?)
    fig = plt.figure(figsize=(1920/96.0, 1080/96.0), dpi=96)
    # fig = plt.figure(figsize=(15.1, 8.5), dpi=127)
    # fig.set_size_inches(15.1, 8.5)
    fig.suptitle('Effect of online learning adjustments', fontsize=20)
    
    step = int((len(item)-1) / 6)
    for i in range(step, len(item), step):
        plt.subplot(2, 3, int(i/step))

        plt.scatter(item[i].reB, item[i].ec, color='red', s=25, label='Before learning')
        plt.scatter(item[i].reA, item[i].ec, color='green', s=25, label='After learning')
        plt.scatter(item[i].pl, item[i].ec, color='orange', s=25, label='ActivForms')
        plt.legend()

        adjustLabels()
        adjustXAxis((minPL, maxPL))


    outputDir = os.path.join(PLOT_OUTPUT_DIR, 'LearningEffect1Cycle')
    if not(os.path.isdir(outputDir)):
        os.mkdir(outputDir)
    plt.savefig(os.path.join(outputDir, f'effectLearning1Cycle_{graphName}.png'), bbox_inches='tight')
    plt.close()
    # plt.show()


def plotLearningEffect2Cycles(item, minEC, maxEC, graphName):
    dataPacketLoss = [ao.pl for ao in item[0:60]]
    dataRegression = [ao.reB for ao in item[0:60]]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    fig = plt.figure(figsize=(1920/96.0, 1080/96.0), dpi=96)
    fig.suptitle('Effect of online learning adjustments over 2 cycles', fontsize=20)
    
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


    outputDir = os.path.join(PLOT_OUTPUT_DIR, 'LearningEffect2Cycles')
    if not(os.path.isdir(outputDir)):
        os.mkdir(outputDir)
    plt.savefig(os.path.join(outputDir, f'LearningEffect2Cycles_{graphName}.png'), bbox_inches='tight')

    # plt.show()
    plt.close()


def plotEffectUncertainties(item, graphName):
    outputDir = os.path.join(PLOT_OUTPUT_DIR, 'EffectUncertainties')
    if not(os.path.isdir(outputDir)):
        os.mkdir(outputDir)

    dataPacketLoss = [ao.pl for ao in item[0:60]]
    dataRegression = [ao.reB for ao in item[0:60]]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    dataEnergyConsumption = [ao.ec for ao in item[0:60]]
    minEC, maxEC = min(dataEnergyConsumption), max(dataEnergyConsumption)

    fig = plt.figure(figsize=(1920/96.0, 1080/96.0), dpi=96)
    fig.suptitle('Effect of uncertainties on adaptation option', fontsize=18)

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
    

    plt.savefig(os.path.join(outputDir, f'EffectUncertainties_{graphName}.png'), bbox_inches='tight')
    plt.close()
    # plt.show()



        

def plotRegressionPredictions(item, minEC, maxEC):
    # TODO add save to file here
    dataPacketLoss = [ao.pl for ao in item]
    dataRegression = [ao.reB for ao in item]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)

    plt.figure()

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

    plt.show()


def plotLearningEvolution(item, minEC, maxEC, graphName):
    outputDir = os.path.join(PLOT_OUTPUT_DIR, 'LearningEvolution')
    if not(os.path.isdir(outputDir)):
        os.mkdir(outputDir)

    dataPacketLoss = [ao.pl for ao in item]
    dataRegression = [ao.reB for ao in item]
    minPL, maxPL = getMinMax(dataPacketLoss, dataRegression)


    fig = plt.figure(figsize=(1920/96.0, 1080/96.0), dpi=96)
    fig.suptitle('Effect of online learning adjustments (regression)', fontsize=18)

    step = int(len(item) / 3)
    for i in range(3):
        cutOffA, cutOffB = i * step, min((i+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+1)
        ax.set_title(f'Samples [{cutOffA}:{cutOffB}] before online learning')

        for point in subPoints:
            color = 'xkcd:orange' if point.reB >= 10 else 'xkcd:green'
            plt.scatter(point.pl, point.ec, color=color, s=20)

        _, _, minY, maxY = plt.axis()
        plt.plot([10, 10], [minY, maxY], color='red')
        plt.xlabel('Packet Loss (%)')
        plt.ylabel('Energy Consumption (coulomb)')
        adjustXAxis((minPL, maxPL))


    for i in range(3):
        cutOffA, cutOffB = i * step, min((i+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+4)
        ax.set_title(f'Samples [{cutOffA}:{cutOffB}] after online learning')

        for point in subPoints:
            color = 'xkcd:orange' if point.reA >= 10 else 'xkcd:green'
            plt.scatter(point.pl, point.ec, color=color, s=20)

        _, _, minY, maxY = plt.axis()
        plt.plot([10, 10], [minY, maxY], color='red')
        plt.xlabel('Packet Loss (%)')
        plt.ylabel('Energy Consumption (coulomb)')
        adjustXAxis((minPL, maxPL))


    plt.savefig(os.path.join(outputDir, f'LearningEvolution_regression_{graphName}.png'), bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(1920/96.0, 1080/96.0), dpi=96)
    fig.suptitle('Effect of online learning adjustments (classification)', fontsize=18)

    step = int(len(item) / 3)
    for i in range(3):
        cutOffA, cutOffB = i * step, min((i+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+1)
        ax.set_title(f'Samples [{cutOffA}:{cutOffB}] before online learning')

        for point in subPoints:
            color = 'xkcd:orange' if point.clB == 0 else 'xkcd:blue'
            plt.scatter(point.pl, point.ec, color=color, s=20)

        _, _, minY, maxY = plt.axis()
        plt.plot([10, 10], [minY, maxY], color='red')
        plt.xlabel('Packet Loss (%)')
        plt.ylabel('Energy Consumption (coulomb)')
        adjustXAxis((minPL, maxPL))


    for i in range(3):
        cutOffA, cutOffB = i * step, min((i+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+4)
        ax.set_title(f'Samples [{cutOffA}:{cutOffB}] after online learning')

        # print(f'Amount of samples accepted in subPoints: {sum(1 if a.clA == 1 else 0 for a in subPoints)}')
        for point in subPoints:
            color = 'xkcd:orange' if point.clA == 0 else 'xkcd:blue'
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
    


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Output directory not provided as commandline argument, using './GraphOutputs' by default")
        PLOT_OUTPUT_DIR = os.path.join('GraphOutputs')
    else:
        PLOT_OUTPUT_DIR = sys.argv[1]

    if os.path.isdir(PLOT_OUTPUT_DIR):
        shutil.rmtree(PLOT_OUTPUT_DIR)

    os.mkdir(PLOT_OUTPUT_DIR)

    analyseUncertainties()

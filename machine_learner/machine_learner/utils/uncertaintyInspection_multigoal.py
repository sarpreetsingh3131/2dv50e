import numpy as np
import json
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import functools
import sys
import os
import shutil
from math import ceil
from dataLoader import loadData
from printUtils import printProgressBar

PLOT_OUTPUT_DIR = ''


def analyseUncertainties(filename = os.path.join('machine_learner', 'collected_data', 'overall_adaptation_options.json')):
    # pathFile = os.path.join('machine_learner', 'collected_data', 'overall_adaptation_options.json')

    adapResults = loadData(filename)

    
    # Get the minimum and maximum value for energy consumption over all configurations (used in graphs)
    ecResults = [[ao.ec for ao in adapResult] for adapResult in adapResults]
    # flatten the list
    ecResults = [item for sublist in ecResults for item in sublist]
    # minEC, maxEC = min(ecResults), max(ecResults)

    print()


    # ===================
    # Plotting of results
    # ===================

    # Plots for all adaptation options
    # for i in range(len(adapResults)):
    #     plotSingleConfiguration(adapResults[i], f'all{i+1}')
    #     # plotLearningEvolution(adapResults[i], f'all{i+1}')
    #     printProgressBar(i+1, len(adapResults), prefix="Progress all graphs:", suffix='Complete', length=30)

    # indices = [1, 2, 3, 4, 5, 55, 58, 60, 61, 72, 78, 112, 164, 168, 170] # Classification DeltaIoTv1
    # indices = [1, 2, 3, 37, 38, 41, 42, 44, 60, 61, 64, 65, 66, 71, 73, 83] # Regression DeltaIoTv1
    # indices = [367,769,2122,2179,2206] # Classification DeltaIoTv2
    # indices = [] # Regression DeltaIoTv2

    # indices = [3,31,262,646,742,727] # Difficult configurations classification DeltaIoTv2
    indices = [31]
    index = 0
    printProgressBar(index, len(indices), prefix="Progress specific graphs:", suffix='Complete', length=30)
    for i in indices:
        plotSingleConfiguration(adapResults[i-1], f'configuration_{i}')
        # plotLearningEvolution(adapResults[i-1], f'configuration_{i}')
        index += 1
        printProgressBar(index, len(indices), prefix="Progress specific graphs:", suffix='Complete', length=30)


    # indices = [173, 174, 363, 365, 366, 384, 392, 411, 741, 744, 749, 750, 751, 752, 753, 756, 759, 761, 762, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 830, 831, 832, 833, 834, 852, 854, 855, 857, 858, 860, 861, 888, 915, 933, 939, 942, 969, 1134, 1542, 1544, 1545, 1546, 1547, 1548, 2095, 2096, 2097, 2111, 2112, 2113, 2114, 2115, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2177, 2178, 2193, 2194, 2195, 2196, 2202, 2203, 2204, 2205, 2258, 2259, 2284, 2285, 2286, 2547, 2555, 2556, 3114, 3122, 3123, 3222, 3254, 3255, 3256, 3257, 3258, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3284, 3285, 3292, 3293, 3294, 3298, 3299, 3300, 3301, 3302, 3303, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3474, 3482, 3489, 3490, 3491, 3492, 3506, 3507, 3509, 3510, 3513, 3514, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3546, 3550, 3551, 3552, 3553, 3554, 3555, 3560, 3561, 3562, 3563, 3564, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3577, 3578, 3579, 3580, 3581, 3582, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3662, 3663, 3669, 3671, 3672, 3681, 3735, 3742, 3743, 3744, 3751, 3752, 3753, 3758, 3759, 3760, 3761, 3762, 3766, 3767, 3768, 3769, 3770, 3771, 3807, 3813, 3814, 3815, 3816, 3821, 3822, 3823, 3824, 3825, 3830, 3831, 3832, 3833, 3834, 3837, 3838, 3840, 3841, 3842, 3843, 3847, 3848, 3849, 3850, 3851, 3852, 3922, 3923, 3930, 3931, 3932, 3933, 3995, 4002, 4004, 4005, 4011, 4012, 4013, 4014, 4050, 4059, 4067, 4068, 4093, 4094, 4095]
    # # indices = [363, 3523, 3524, 3578, 3587]
    # indices = [741]
    # index = 0
    # print()
    # printProgressBar(index, len(indices), prefix="Progress specific graphs:", suffix='Complete', length=30)
    # for i in indices[::-1]:
    #     plotEffectUncertaintiesPLLA(adapResults[i], f'configuration_{i}')
    #     index += 1
    #     printProgressBar(index, len(indices), prefix="Progress specific graphs:", suffix='Complete', length=30)
    print()




def plotMetaGrid():
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(1,1,1)

    # Line for packet loss
    plt.plot([10, 10], [0, 10], color='red')
    # Line for latency
    plt.plot([0, 20], [5,5], color='red')

    step = 0.01
    x1 = np.arange(0, 10, step)
    x2 = np.arange(10 + step, 20 + step, step)
    
    a = 0.75
    ax.fill_between(x1, 0, 5, facecolor='xkcd:blue', alpha = a)
    ax.fill_between(x2, 0, 5, facecolor='xkcd:lime green', alpha = a)
    ax.fill_between(x1, 5, 10, facecolor='xkcd:lavender', alpha = a)
    ax.fill_between(x2, 5, 10, facecolor='xkcd:orange', alpha = a)
    
    adjustLabels('Packet Loss (%)', 'Latency (%)')
    plt.show()





def plotEffectUncertainties(configuration, graphName):
    # dataPacketLoss = [ao.pl for ao in configuration[0:60]]
    # minPL, maxPL = min(dataPacketLoss), max(dataPacketLoss)
    dataEnergyConsumption = [ao.ec for ao in configuration[0:60]]
    minEC, maxEC = min(dataEnergyConsumption), max(dataEnergyConsumption)


    mlp.rcParams['font.size'] = 18

    # Plot 10 data points per graph
    for i in range(1):
        # TODO remove
        i = 15

        plt.figure(figsize=(10,10))
        ax = plt.subplot(1, 1, 1)

        # plot the arrows between the points
        for j in range(i * 10, (i * 10) + 9):
            first = configuration[j]
            second = configuration[j+1]
            
            disX = abs(first.pl - second.pl)
            disY = abs(first.ec - second.ec) 
            dis = (disX**2 + disY**2)**.5

            adjX = (disX / dis) * 0.025 * (1 if first.pl < second.pl else -1)
            adjY = (disY / dis) * 0.009 * (1 if first.ec < second.ec else -1)
            ax.add_patch(mpatches.FancyArrowPatch(
                (first.pl + adjX, first.ec + adjY), 
                (second.pl - adjX, second.ec - adjY), 
                mutation_scale=20)
            )

        # plot the points
        for j in range(i * 10, (i + 1) * 10):
            plt.scatter(configuration[j].pl, configuration[j].ec, color='green', s=150, marker='o', edgecolors='black')

        plt.xlabel('Packet loss (%)')
        plt.ylabel('Energy consumption (Coulomb)')
        # x1, x2, y1, y2 = plt.axis()
        plt.plot([10, 10], [66,69], color='red')
        plt.plot([10, 10], [minEC-0.1,maxEC+0.1], color='red')
        # plt.axis([x1-0.2, x2+0.2, y1, y2])
        plt.axis([7, 12, 66,69])
        # plt.axis([minPL-0.5, maxPL+0.5, minEC-0.1, maxEC+0.1])

        outputDir = createOrGetDir('EffectUncertainties')
        # plt.savefig(os.path.join(outputDir, f'EffectUncertainties_{graphName}[{(i*10)+1}-{((i+1)*10+1)}].pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(outputDir, f'EffectUncertainties_{graphName}_Cycles[{(i*10)+1}-{((i+1)*10)}].png'), bbox_inches='tight')
        plt.close()
    

def plotEffectUncertaintiesPLLA(configuration, graphName):
    # dataPacketLoss = [ao.pl for ao in configuration[0:60]]
    # minPL, maxPL = min(dataPacketLoss), max(dataPacketLoss)
    # dataEnergyConsumption = [ao.ec for ao in configuration[0:60]]
    # minEC, maxEC = min(dataEnergyConsumption), max(dataEnergyConsumption)


    mlp.rcParams['font.size'] = 18

    # Plot 10 data points per graph
    for i in range(1):
        # TODO remove
        i = 15

        plt.figure(figsize=(10,10))
        ax = plt.subplot(1, 1, 1)

        # plot the arrows between the points
        for j in range(i * 10, (i * 10) + 9):
            first = configuration[j]
            second = configuration[j+1]
            
            disX = abs(first.pl - second.pl)
            disY = abs(first.la - second.la) 
            dis = (disX**2 + disY**2)**.5
            if dis != 0:
                adjX = (disX / dis) * 0.025 * (1 if first.pl < second.pl else -1)
                adjY = (disY / dis) * 0.009 * (1 if first.la < second.la else -1)
            else:
                adjX = 0
                adjY = 0
            ax.add_patch(mpatches.FancyArrowPatch(
                (first.pl + adjX, first.la + adjY), 
                (second.pl - adjX, second.la - adjY), 
                mutation_scale=20)
            )

        # plot the points
        for j in range(i * 10, (i + 1) * 10):
            if j == i*10:
                plt.scatter(configuration[j].pl, configuration[j].la, color='green', s=200, marker='*', edgecolors='black')
            else:
                plt.scatter(configuration[j].pl, configuration[j].la, color='green', s=150, marker='o', edgecolors='black')

        plt.xlabel('Packet loss (%)')
        plt.ylabel('Latency (%)')
        # x1, x2, y1, y2 = plt.axis()
        plt.plot([10,10], [0,8], color='red')
        plt.plot([7,12], [5,5], color='red')
        plt.axis([7, 12, 0, 8])

        outputDir = createOrGetDir('EffectUncertaintiesPLLA')
        # plt.savefig(os.path.join(outputDir, f'EffectUncertainties_{graphName}[{(i*10)+1}-{((i+1)*10+1)}].pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(outputDir, f'EffectUncertainties_{graphName}_Cycles[{(i*10)+1}-{((i+1)*10)}].pdf'), bbox_inches='tight')
        plt.close()



def plotLearningEvolution(item, graphName):
    outputDir = createOrGetDir('LearningEvolution')

    dataPacketLoss = [ao.pl for ao in item]
    dataLatency = [ao.la for ao in item]

    minPL, maxPL = min(dataPacketLoss), max(dataPacketLoss)
    minLA, maxLA = min(dataLatency), max(dataLatency)

    # Classification
    initialiseFigure('Effect of online learning adjustments (1 configuration, classification)')
    step = ceil(len(item) / 3)
    for i in range(6):
        cutOffA, cutOffB = (i%3) * step, min(((i%3)+1) * step, len(item))
        subPoints = item[cutOffA : cutOffB]
        ax = plt.subplot(2, 3, i+1)
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
                color = 'xkcd:lavender'

            # TODO: could vary size/color opacity here to indicate how good the energy consumption of this point is
            plt.scatter(point.pl, point.la, color=color, s=20)
            
        # Line for packet loss
        plt.plot([10, 10], [minLA, maxLA], color='red')
        # Line for latency
        plt.plot([minPL, maxPL], [5,5], color='red')

        adjustLabels('Packet Loss (%)', 'Latency (%)')
        adjustXAxis((minPL, maxPL))
        adjustYAxis((minLA, maxLA))

        if i == 1:
            plt.legend(
                bbox_to_anchor=(-0.5, 1.13, 2., .122),
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

    plt.savefig(os.path.join(outputDir, f'{graphName}.pdf'), bbox_inches='tight')
    # plt.savefig(os.path.join(outputDir, f'LearningEvolution_classification_{graphName}.pdf'), bbox_inches='tight')
    plt.close()


def plotSingleConfiguration(configuration, graphName):
    outputDir = createOrGetDir('SingleConfigurationsPdf')

    dataPacketLoss = [ao.pl for ao in configuration]
    dataLatency = [ao.la for ao in configuration]

    minPL, maxPL = min(dataPacketLoss), max(dataPacketLoss)
    minLA, maxLA = min(dataLatency), max(dataLatency)

    # Classification
    mlp.rcParams['font.size'] = 18
    plt.figure(figsize=(14,10))

    for point in configuration:
        valToComp = point.clB
        color = 'xkcd:orange'
        if valToComp == 3:
            color = 'xkcd:blue'
        elif valToComp == 2:
            color = 'xkcd:emerald green'
        elif valToComp == 1:
            color = 'xkcd:lavender'

        # TODO: could vary size/color opacity here to indicate how good the energy consumption of this point is
        plt.scatter(point.pl, point.la, color=color, s=20)
        
    # Line for packet loss
    plt.plot([10, 10], [0, 12], color='red')
    # Line for latency
    plt.plot([5, 25], [5,5], color='red')

    adjustLabels('Packet loss (%)', 'Latency (%)')
    adjustXAxis((minPL, maxPL))
    adjustYAxis((minLA, maxLA))

    plt.axis([5, 25, 0, 12])

    plt.legend(
        bbox_to_anchor=(0, 1, 1., 0),
        ncol=4,
        mode='expand',
        fontsize='large',
        handles=[
            mpatches.Patch(color='xkcd:blue', label='$C_3$'),
            mpatches.Patch(color='xkcd:emerald green', label='$C_2$'),
            mpatches.Patch(color='xkcd:lavender', label='$C_1$'),
            mpatches.Patch(color='xkcd:orange', label='$C_0$')
        ]
    )

    plt.savefig(os.path.join(outputDir, f'{graphName}.pdf'), bbox_inches='tight')
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
    # fig = plt.figure(figsize=(16,9), dpi=300)
    fig.suptitle(title, fontsize=fontsize, y=1.014)
    return fig



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Output directory not provided as commandline argument, using './GraphOutputs' by default")
        PLOT_OUTPUT_DIR = os.path.join('GraphOutputs')
    else:
        PLOT_OUTPUT_DIR = sys.argv[1]

    # if os.path.isdir(PLOT_OUTPUT_DIR):
    #     shutil.rmtree(PLOT_OUTPUT_DIR)

    # os.mkdir(PLOT_OUTPUT_DIR)

    # plotMetaGrid()
    # sys.exit(0)

    if len(sys.argv) == 3:
        analyseUncertainties(sys.argv[2])
    else:
        analyseUncertainties()

from util import getData, constructDataWithSelectedFeatures
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# the file to be analysed is set in the util.py file


P_VALUE = 0.05

def main():
    
    data = getData()

    features = np.array(data["features"])
    targets = np.array(data["targets"])

    print(features.shape)
    print(targets.shape)

    tree = ExtraTreesClassifier(n_estimators=100,random_state=10)

    tree.fit(features,targets)

    importances = tree.feature_importances_


    load = []
    power = []
    dist = []
    snr = []
    loads = []
    powers = []
    dists = []
    snrs = []
    ticks = []

    fig = plt.figure()

    i=0
    ticks.append(1)
    while (i < features.shape[1]):

        load.append(importances[i])
        loads.append(i+1)
        power.append(importances[i+1])
        powers.append(i+2)
        dist.append(importances[i+2])
        dists.append(i+3)
        snr.append(importances[i+3])
        snrs.append(i+4)

        if (i == 5*4 or i == 8* 4 + 3 or i == 10*4 + 6):

            power.append(importances[i+4])
            powers.append(i+5)
            dist.append(importances[i+5])
            dists.append(i+6)
            snr.append(importances[i+6])
            snrs.append(i+7)

            i += 7
        else:
            i += 4
        ticks.append(i+1)
    
    plt.bar(loads, load, color='r', label = 'Importance mote load')
    plt.bar(powers, power, color='b', label="Importance link power")
    plt.bar(dists, dist, color='g', label="Importance link distribution")
    plt.bar(snrs, snr, color='black', label="Importance link SNR")

    #sf = MinMaxScaler().fit_transform(features)
    #print(sf.shape)
    c, p = f_classif(features, targets)

    yes = []
    no = []
    for i in range(0,len(p)):
        if p[i] < P_VALUE:
            no.append(i)
        else:
            yes.append(i)

    yesp = yes.copy()
    nop = no.copy()
    for i in range(0,len(yesp)):
        yesp[i] = yesp[i] + 1
    for i in range(0,len(nop)):
        nop[i] = nop[i] + 1
    
    plt.plot(yesp, importances[yes], 'rx', label="Not rejected by f_classif")
    plt.plot(nop, importances[no], 'y*', label="Rejected by f_classif\n at significance level "+str(P_VALUE))
    
    plt.xticks(ticks)
    plt.title("Feature selection packet loss and latency. \n (gap in xtick represents one mote and its links)")
    plt.legend()
    plt.show()
    fig.savefig(os.path.join("fsFigures",'featureSelection.eps'), format='eps', dpi=1200)
    fig.savefig(os.path.join("fsFigures",'featureSelection.svg'), format='svg', dpi=1200)
    fig.savefig(os.path.join("fsFigures",'featureSelection.png'), format='png', dpi=1200)

    x = "asdf"
    while( x != 'y' and x != 'n' ):
        print("Features selected by f_classif: ", no)
        x = input("Do you want to create data with the features selected by f_classif? [y/n]\n")
    if (x == 'y'):
        constructDataWithSelectedFeatures(no)

    x = "asdf"
    while( x != 'y' and x != 'n' ):
        print("Features selected by f_classif for 456Cycles20Dist: ", [3, 7, 11, 15, 19, 22, 23, 25, 26, 30, 34, 35, 37, 38, 40, 41, 45, 48, 49, 51, 52, 53, 56, 60, 64])
        x = input("Do you want to create data with the features selected by f_classif for 456Cycles20Dist? [y/n]\n")
    if (x == 'y'):
        constructDataWithSelectedFeatures([3, 7, 11, 15, 19, 22, 23, 25, 26, 30, 34, 35, 37, 38, 40, 41, 45, 48, 49, 51, 52, 53, 56, 60, 64])
    
 


"""
    i = 0
    while i <= len(targets):
        j = i
        i += 4
        if (j == 6*4 or j == 9* 4 + 3 or j == 11*4 + 6):
            i += 3
        

        if (j == 0):

            
            plt.bar(j+1, importances[j], 'r', label="Mote load")
            plt.bar(j+2, importances[j +1], 'g', label="Link power")
            plt.bar(j+3, importances[j+2], 'b', label="Link distribution")
            plt.bar(j+4, importances[j+3], 'p', label="Link SNR")
        else:
            plt.bar(j+1, importances[j], 'r')
            plt.bar(j+2, importances[j +1], 'g')
            plt.bar(j+3, importances[j+2], 'b')
            plt.bar(j+4, importances[j+3], 'p')
        
        if (j == 6*4 or j == 9* 4 + 3 or j == 11*4 + 6):

            plt.bar(j+5, importances[j+4], 'g/')
            plt.bar(j+6, importances[j+5], 'b/')
            plt.bar(j+7, importances[j+6], 'p/')
"""
    
        


    

main()
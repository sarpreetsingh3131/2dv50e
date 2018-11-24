from util import getData
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

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


    plt.subplots()

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
    
    plt.show()
    input()
 

        


    

main()
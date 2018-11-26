import os
from util import getDataWithSelectedFeatures
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt



def main():

    data = getDataWithSelectedFeatures()
    features = data["features"]
    targets = data["targets"]

    # the robust scaler works better for outliers
    # I use the default parameters, but you could adjust them and leave them in the scaler list
    # and the algorithm will still work.
    scalers = [ sp.MaxAbsScaler(), sp.MinMaxScaler(), sp.RobustScaler(), sp.StandardScaler(), None]

    #TODO: should we transform the data with the transformers?

    dimReduction = [None]
    # TODO: add the dim reduction and classifiers
    classifiers = [None]

    for scaler in scalers:

        for dimReducer in dimReduction:

            for classifier in classifiers:

                if (scaler is None):
                    scaled = features
                else:
                    scaled = scaler.fit_transform(features)

                #TODO: dim reduction


    #if scaler is None:


main()
"""A file used for preparing the targets of new data for machine learning

To add ml types, add a function for it and return the name of the folder
and the formatted data.
Than add the function to the mlTypes variable.
The directory and file will be created automatically.

"""
import json
import os
from os.path import realpath, join, dirname
import sys

if len(sys.argv) == 2:
    RDATA_FILE = sys.argv[1]
else:
    #TODO: change this to the name of the raw data file (without trailing .json)
    RDATA_FILE = "456Cycles20Dist"


PACKETLOSS_GOAL = 10.0
LATENCY_GOAL = 0.0135

def plRegression(targets):

    formatted = []

    for entry in targets:
        formatted.append(entry["packetLoss"])

    return ["plRegression", formatted]

def plClassification(targets):

    formatted = []

    for entry in targets:
        if float(entry["packetLoss"]) <= PACKETLOSS_GOAL:
            formatted.append(1)
        else:
            formatted.append(0)
    
    return ["plClassification", formatted]


def plLaClassification(targets):

    formatted = []

    # I use biflags per goal, rightmost bit is packetloss and to the left is latency bit.
    for entry in targets:
        cl = 0
        if float(entry["packetLoss"]) <= PACKETLOSS_GOAL:
            cl += 1
        if float(entry["latency"]) <= LATENCY_GOAL:
            cl += 2
        
        formatted.append(cl)

    return ["plLaClassification", formatted]



# no exception catching because I want to read error
def main():

    #TODO: add the function of the machine learner types you want to format for
    mlTypes = [plLaClassification]

    # locate file
    rawDataPath = realpath(__file__)
    for i in range(1,4): 
        rawDataPath = dirname(rawDataPath)

    rawDataPath = join(rawDataPath, "data", "rawSimData", RDATA_FILE + ".json")

    # load data
    with open(rawDataPath, 'r') as f:
        unformatted = json.load(f)


    # for all ml types
    for mlType in mlTypes:

        # copy features
        formatted = {}
        formatted["features"] = unformatted["features"]

        # format targets
        resp = mlType(unformatted["targets"])
        formatted["targets"] = resp[1]

        # make directory 
        if not os.path.exists(resp[0]):
            os.makedirs(resp[0])

        # write data to appropriate file
        with open(os.path.join(resp[0], RDATA_FILE + ".json"), 'w+') as f:
            json.dump(formatted, f)

main()
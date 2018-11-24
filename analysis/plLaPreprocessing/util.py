from os.path import join, realpath, dirname
import json


def getData():
    RDATA_FILE = "456Cycles20Dist"

    # locate file
    rawDataPath = realpath(__file__)
    for i in range(1,4): 
        rawDataPath = dirname(rawDataPath)

    rawDataPath = join(rawDataPath, "data", "formattedData", "plLaClassification", RDATA_FILE + ".json")

    with open(rawDataPath, 'r') as f:
        data = json.load(f)

    return data     



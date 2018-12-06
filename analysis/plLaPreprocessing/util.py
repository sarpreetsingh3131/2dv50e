from os.path import join, realpath, dirname, exists
import os
import json

# change this filename to analyze a different file from 
# the formatted plLaClassification
# You should just be able to run featureSelection first,
# and if you answer yes to the question to construct data with selected features
# you can just execute modelSelection.py
RDATA_FILE = "456Cycles20Dist"

def getData():

    # locate file
    rawDataPath = realpath(__file__)
    for i in range(1,4): 
        rawDataPath = dirname(rawDataPath)

    rawDataPath = join(rawDataPath, "data", "formattedData", "plLaClassification", RDATA_FILE + ".json")

    with open(rawDataPath, 'r') as f:
        data = json.load(f)

    return data     

def constructDataWithSelectedFeatures( selectedFeatures ):

    path = realpath(__file__)
    path = join(dirname(path), "dataWithSelectedFeatures")
    if not exists(path):
        os.makedirs(path)
    
    data = getData()

    fs = {}
    fs["targets"] = data["targets"]
    fs["features"] = []
    r = []

    for row in data["features"]:
        r = []
        for i in selectedFeatures:
            r.append(row[i])
        
        fs["features"].append(r)


    with open( join(path, RDATA_FILE + ".json"), 'w') as f:
        json.dump(fs,f)

def getDataWithSelectedFeatures():

    path = realpath(__file__)
    path = join(dirname(path), "dataWithSelectedFeatures", RDATA_FILE + ".json")

    with open( path, 'r') as f:
        data = json.load(f)

    return data
    

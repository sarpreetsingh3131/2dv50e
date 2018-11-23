import json
from os.path import join, dirname, realpath
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import pandas as pd

# get the file from data/rawSimData

datafile = "456Cycles20Dist"
plLimit = 10

rawDataPath = realpath(__file__)
for i in range(1,4): 
    rawDataPath = dirname(rawDataPath)

rawDataPath = join(rawDataPath, "data", "rawSimData", datafile + ".json")

with open(rawDataPath, 'r') as f:
    data = json.load(f)

plList = []
laList = []
for entry in data["targets"]:
    plList.append(entry["packetLoss"])
    laList.append(entry["latency"])

cumPl = 0
for p in plList:
    if p <= plLimit:
        cumPl += 1
cumPl /= len(plList)

pl = np.array(plList)
la = np.array(laList)

df = pd.Series(la)
q = df.quantile(cumPl)

fig = plt.figure()
#plPlot = fig.add_subplot(3,2,1)
#laPlot = fig.add_subplot(3,2,2)
#heat = fig.add_subplot(2,1,2)
plPlot = plt.subplot2grid((3,2), (0,0))
laPlot = plt.subplot2grid((3,2), (0,1))
heat = plt.subplot2grid((3,2), (1,0), colspan=2)
heat2 = plt.subplot2grid((3,2), (2,0))
la2 = plt.subplot2grid((3,2), (2,1))

# amount of bins is from the formula of Sturges
nbOfBins = math.ceil(math.log2(pl.size) + 1)
plPlot.hist(pl, bins=nbOfBins*2)
plPlot.set_title('Packetloss distribution')
plPlot.axvline(x=plLimit, color='r')
laPlot.hist(la, bins=nbOfBins*2)
laPlot.set_title('Latency distribution \n'+ "{:.4f}".format(q))
laPlot.axvline(x=q, color='r')

heat.hist2d(pl, la, bins=nbOfBins*2)
heat.axvline(x=plLimit, color='r')
heat.axhline(y=q, color='r')
heat.set_xlabel='packetloss'
heat.set_ylabel='latency'
heat.set_title("Heatmap packetloss and latency")

la2.hist(la, bins=nbOfBins*2, range=(0,0.1))
la2.set_title('Latency distribution \n'+ "{:.4f}".format(q))
la2.axvline(x=q, color='r')

heat2.hist2d(pl, la, bins=nbOfBins*2, range=[[np.amin(pl),np.amax(pl)],[0,0.015]])
heat2.axvline(x=plLimit, color='r')
heat2.axhline(y=q, color='r')
heat2.set_xlabel='packetloss'
heat2.set_ylabel='latency'
heat2.set_title("Heatmap packetloss and latency")


fig.tight_layout()
fig.savefig('determine.eps', format='eps', dpi=1200)
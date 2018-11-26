import sys
import json
import numpy as np

class AdaptationResult:
    def __init__(self, ec, pl, clB, reB, clA, reA):
        self.ec = ec    # ec = Energy consumption (as evaluated by uppaal)
        self.pl = pl    # pl = packet loss (as evaluated by uppaal)
        self.clB = clB  # clB = classification prediction before online training adjustment
        self.reB = reB  # clA = classification prediction after online training adjustment
        self.clA = clA  # reB = regression prediction before online training adjustment
        self.reA = reA  # reA = regression prediction after online training adjustment

    # Returns a tuple about the correctness (before online learning) of
    # the prediction from classification and regression respectively
    def isPredictedRight(self):
        # For classification, check if the predicted class is correct for the evaluated packet loss
        # For regression, make sure the sign of the predicted and actual packet loss value -10 is equal
        return (True if (self.pl < 10) == self.clB else False, ((self.pl - 10) < 0) == ((self.reB - 10) < 0))

    # Negation of isPredictedRight method
    def isPredictedWrong(self):
        tmp = self.isPredictedRight()
        return (not tmp[0], not tmp[1])

    def isPredictedWrongPerSide(self):
        if self.pl < 10:
            # Analyse if the prediction was right left to the line
            return ((self.clB != 1, 0), (self.reB >= 10, 0))
        else:
            return ((0, self.clB != 0), (0, self.reB < 10))


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

    def getAmtResultsPerSide(self):
        amtLeft = sum([i.pl < 10 for i in self.results])
        return (amtLeft, len(self.results) - amtLeft)


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

    # Gets the amount of wrong predictions for classification and regression respectively
    # Returns a tuple: (#err classification, #err regression)
    def getAmtOfWrongPredictions(self):
        predicts = [i.isPredictedWrong() for i in self.results]
        # Count the cycles where the prediction of the respective learners went wrong
        return tuple([sum(x) for x in zip(*predicts)])


    # Get the amount of wrong predictions for each side of the cutoff line
    # Returns: ((#err class left to line, #err class right to line),(#err regr left to line, #err regr right to line))
    def getWrongPredictionsPerSide(self):
        res = [i.isPredictedWrongPerSide() for i in self.results]
        return ((sum([i[0][0] for i in res]), sum([i[0][1] for i in res])),(sum([i[1][0] for i in res]), sum([i[1][1] for i in res])))

    # Gets the confusion matrix for this configuration.
    # Returns a dictionary, with mappings [TruePositives, TrueNegatives, FalsePositives, FalseNegatives]
    def getConfusionMatrix(self, technique='classification'):
        res = {'TruePositives': 0, 'TrueNegatives': 0, 'FalsePositives': 0, 'FalseNegatives': 0}
        if technique == 'classification':
            predictedGood = lambda x: x.clB == 1
        elif technique == 'regression':
            predictedGood = lambda x: x.reB < 10

        for i in self.results:
            if i.pl < 10:
                res['TruePositives' if predictedGood(i) else 'FalseNegatives'] += 1
            else:
                res['FalsePositives' if predictedGood(i) else 'TrueNegatives'] += 1
        
        return res



def loadData(pathFile):
    try:
        data = json.load(open(pathFile))
    except Exception:
        print(f'Could not open file at location \'{pathFile}\'')
        sys.exit(1)
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

            # NOTE Amount of learning cycles hardcoded for now
            # TODO maybe move indication of learning cycle to the data itself
            if i > 29:
                adapResults[ai].addResult(ec, pl, clB, reB, clA, reA)

    return adapResults

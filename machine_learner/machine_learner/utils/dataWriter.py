import json
import os



def appendDataMultiGoal(filename, cycle, training, pl, ec, la, clB, clA, rePLB, rePLA, reLAB, reLAA):
    if cycle == 1:
        if os.path.exists(filename):
            os.remove(filename)
        data = []
    else:
        with open(filename, 'r') as f:
            data = json.load(f)
    

    data.append({
        'cycle' : cycle,
        'training': 'true' if training else 'false',
        'adaptationOptions' : {
            'packetLoss' : pl,
			'energyConsumption' : ec,
            'latency' : la,
			'classificationBefore' : clB,
			'classificationAfter' : clA,
			'regressionPLBefore' : rePLB,
			'regressionPLAfter' : rePLA,
            'regressionLABefore' : reLAB,
            'regressionLAAfter' : reLAA
        }
    })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=1)

def appendDataSingleGoal(filename, cycle, training, pl, ec, clB, clA, rePLB, rePLA):
    appendDataMultiGoal(filename, cycle, training, pl, ec, [], clB, clA, rePLB, rePLA, [], [])

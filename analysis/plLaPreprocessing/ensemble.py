from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
import matplotlib as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from util import getDataWithSelectedFeatures
import time
import numpy as np
from sklearn.model_selection import cross_val_score
import warnings

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


    NB_ADAPTION_PER_CYCLE=216
    data = getDataWithSelectedFeatures()
    scaler = StandardScaler()
    rn = 10
    IT = 1
    to = None
    ensLearner = VotingClassifier(
        estimators=[
            ('SGD-h-e--2', SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
            ('PA-I--3', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.001, tol=to, random_state=rn)),
            ('SGD-h-l2--2', SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
            ('PA-II--4', PassiveAggressiveClassifier(loss='squared_hinge', max_iter=IT, C=0.0001, tol=to, random_state=rn)),
            ('PA-I--4', PassiveAggressiveClassifier(loss='hinge', max_iter=IT, C=0.0001, tol=to, random_state=rn)),
            ('SGD-h-l1--2', SGDClassifier(loss='hinge', penalty='l1', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
            ('SGD-l-e--1', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
            ('SGD-mh-e--1', SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
            ('SGD-l-e--2', SGDClassifier(loss='log', penalty='elasticnet', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
            ('SGD-mh-l2--1', SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
            ('SGD-l-l2--2', SGDClassifier(loss='log', penalty='l2', alpha=0.01, max_iter=IT, tol=to, random_state=rn)),
            ('SGD-h-l2--1', SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, max_iter=IT, tol=to, random_state=rn)),
        ]
    )
    features = data["features"]
    targets = data["targets"]

    # Make data ready
    kf = 10

    skf = StratifiedKFold(n_splits=kf, shuffle=True, random_state=10)

    #(trainIndeces, testIndices) = skf.split(features,targets)
    ks = skf.split(features,targets)

    trainSetsF = []
    trainSetsT = []
    testSetsF = []
    testSetsT = []

    tsf = []
    tst = []
    tesf = []
    tesk = []

    for (tr,te) in ks:
        tsf = []
        tst = []
        tesf = []
        tesk = []
        for i in tr:
            tsf.append(features[i])
            tst.append(targets[i])
        for i in te:
            tesf.append(features[i])
            tesk.append(targets[i])
        trainSetsF.append(tsf.copy())
        trainSetsT.append(tst.copy())
        testSetsF.append(tesf.copy())
        testSetsT.append(tesk.copy())



    #for dimReducer in dimReduction:
    trainSetsFs = []
    testSetsFs = []

    for i in range(len(trainSetsF)):
        trainSetsFs.append( scaler.fit_transform(trainSetsF[i].copy()))
    for i in range(len(testSetsF)):
        testSetsFs.append(scaler.transform(testSetsF[i].copy()))


    start = time.time()

    a = []
    faults = [0,0,0,0]

    for trainF, trainT, testF, testT in zip(trainSetsFs, trainSetsT, testSetsFs, testSetsT):

        m = 0

        fr = [0,0,0,0]

                #if (not(scaler[1] is None)):
                    #trainF = scaler[1].fit_transform(trainF)
                    #testF = scaler[1].transform(testF)

                # try to partial fit everything

                #for caseF, caseT in zip(trainF, trainT):
                #    caseT = np.array([caseT])
                #    classifier[1].partial_fit(X=caseF.reshape(1,-1), y=caseT, classes=np.array([0,1,2,3]))
        #TODO: partial fit this
        #ensLearner.fit(trainF, trainT)
        for i in range(len(trainF)//NB_ADAPTION_PER_CYCLE-1):
            for learner in ensLearner.estimators:
                learner[1].partial_fit(
                    X=trainF[i*NB_ADAPTION_PER_CYCLE:(i+1)*NB_ADAPTION_PER_CYCLE-1],
                    y=trainT[i*NB_ADAPTION_PER_CYCLE:(i+1)*NB_ADAPTION_PER_CYCLE-1],
                    classes=[0,1,2,3]
                    )
                #print("pf gelukt "+str(len(targets)))
                #print(str(i))


                # I wont use score here, because it returns a harsh
                # metric which will always be low.
                # I will do it manually

        #pred = ensLearner.predict(testF)
        pred = []
        for i in range(len(testF)):
            l = []
            max = 0
            for i in range(kf):
                l.append(0)
            for learner in ensLearner.estimators:
                c = np.array(testF[i]).reshape(1,-1)
                j = learner[1].predict(c)
                l[j[0]] += 1
                if l[j[0]] > max:
                    max = j[0]
            pred.append(max)

        for p,t in zip(pred, testT):
            if (p == t):
                m += 1
            else:
                fr[t] += 1

        valasdf, counts = np.unique(np.array(testT), return_counts=True)
        #print(fr)
        for i in range(len(faults)):
            faults[i] += fr[i] / counts[i]
        m = m / len(testT)
        a.append(m)
                #print(faults)

    me = np.mean(np.array(a))
    va = np.std(np.array(a))

    sec = time.time()-start



            #print(faults)
            #print(counts)
            #for wrong, total in zip(faults, counts):
                #wrong = wrong / total
    for i in range(len(faults)):
        faults[i] = faults[i] / kf

            #print(faults)

    print("Manually:")
    print("Mean "+str(me))
    print("Std "+str(va))
    print("Time "+str(sec))
    print("Fault percentage per class:")
    print(faults)


    scores = cross_val_score(estimator=ensLearner, X=features, y=targets, cv=kf, n_jobs=4)
    print("Cross_val_scores: \n %s" % scores)
    print("Mean "+str(np.mean(scores))+", std "+str(np.std(scores)))



main()
#Manually:
#Mean 0.5414062087101874
#Std 0.12269051079345082
#Time 67.94954991340637
#Fault percentage per class:
#[0.1, 0.9, 1.0, 1.0]
#Cross_val_scores:
# [0.69647751 0.61516597 0.65546645 0.61993706 0.61106599 0.43141436
# 0.5561986  0.65038587 0.55442729 0.61900894]
#Mean 0.6009548041639293, std 0.0696002148933705

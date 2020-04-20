import numpy as np
import pandas as pd

def processingData(list_):
    l = len(list_)
    r90 = 9900
    r100 = 9900
    for k in range(1, l-1):
        if list_[k-1]<0.9 and list_[k]>=0.9:
            r90 = 100*k
        if list_[k-1]<1 and list_[k]==1:
            r100 = 100*k
    
    return r90, r100


hybridT90 = []
hybrid90 = []
hybridT100 = []
hybrid100 = []
randomT90 = []
randomT100 = []
random90 = []
random100 = []
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    HTtemp90 = []
    HTtemp100 = []
    Htemp90 = []
    Htemp100 = []
    RTtemp90 = []
    RTtemp100 = []
    Rtemp90 = []
    Rtemp100 = []
    for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        filename = 'uni' + str(i) + 'res' + str(j)+ 'new test'  + '60ptps.csv' 
        data = pd.read_csv(filename, header = 0)
        result90, result100 = processingData(data.values[0])
        HTtemp90.append(result90)
        HTtemp100.append(result100)

        result90, result100 = processingData(data.values[1])
        Htemp90.append(result90)
        Htemp100.append(result100)

        result90, result100 = processingData(data.values[2])
        RTtemp90.append(result90)
        RTtemp100.append(result100)

        result90, result100 = processingData(data.values[3])
        Rtemp90.append(result90)
        Rtemp100.append(result100)

    hybridT90.append(HTtemp90)
    hybridT100.append(HTtemp100)
    hybrid90.append(Htemp90)
    hybrid100.append(Htemp100)
    randomT90.append(RTtemp90)
    randomT100.append(RTtemp100)
    random90.append(Rtemp90)
    random100.append(Rtemp100)


scoreT90 = []
score90 = []
scoreT100 = []
score100 = []
entropyT90 = []
entropyT100 = []
entropy90 = []
entropy100 = []
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    STtemp90 = []
    STtemp100 = []
    Stemp90 = []
    Stemp100 = []
    ETtemp90 = []
    ETtemp100 = []
    Etemp90 = []
    Etemp100 = []
    for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        filename = r'test(score+entropy)/uni' + str(i) + 'res' + str(j)+ 'score&entropy test '  + '(no threshold)60ptps.csv'
        data = pd.read_csv(filename, header = 0)
        result90, result100 = processingData(data.values[0])
        Stemp90.append(result90)
        Stemp100.append(result100)

        result90, result100 = processingData(data.values[1])
        Etemp90.append(result90)
        Etemp100.append(result100)


        filename = r'testT/uni' + str(i) + 'res' + str(j)+ 'score&entropy test '  + '60ptps.csv'
        data = pd.read_csv(filename, header = 0)
        result90, result100 = processingData(data.values[0])
        STtemp90.append(result90)
        STtemp100.append(result100)

        result90, result100 = processingData(data.values[1])
        ETtemp90.append(result90)
        ETtemp100.append(result100)


    scoreT90.append(STtemp90)
    scoreT100.append(STtemp100)
    score90.append(Stemp90)
    score100.append(Stemp100)
    entropyT90.append(ETtemp90)
    entropyT100.append(ETtemp100)
    entropy90.append(Etemp90)
    entropy100.append(Etemp100)

np.save('matrix/hybrid_threshold90', hybridT90)
np.save('matrix/hybrid90', hybrid90)
np.save('matrix/hybrid_threshold100', hybridT100)
np.save('matrix/hybrid100', hybrid100)
np.save('matrix/random_threshold90', randomT90)
np.save('matrix/random90', random90)
np.save('matrix/random_threshold100', randomT100)
np.save('matrix/random100', random100)
np.save('matrix/score_threshold90', scoreT90)
np.save('matrix/score90', score90)
np.save('matrix/score_threshold100', scoreT100)
np.save('matrix/score100', score100)
np.save('matrix/entropy_threshold90', entropyT90)
np.save('matrix/entropy90', entropy90)
np.save('matrix/entropy_threshold100', entropyT100)
np.save('matrix/entropy100', entropy100)
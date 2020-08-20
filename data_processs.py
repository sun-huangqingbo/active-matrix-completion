import numpy as np
import pandas as pd


def processingData(list_):
    l = len(list_)
    r90 = 0
    r100 = 9900
    for k in range(1, l):
        if list_[k-1]<0.9 and list_[k]>=0.9:
            r90 = 100*(k+1)
            break
    for k in range(1, l):
        if list_[k-1]<1.0 and list_[k]==1.0:
            r100 = 100*(k+1)
            break
    if list_[l-1]<1:
        r100 = 10000
    return r90/10000, r100/10000

mtx90 = np.zeros((4, 4))
mtx100 = np.zeros((4, 4))
mtx90R = np.zeros((4, 4))
mtx100R = np.zeros((4, 4))
file = "feature_version_synthetic_test_rand_cov_active+rand_32.csv"
data = np.genfromtxt(file, delimiter=',')
for i in range(16):
    row = int(i / 4)
    col = i % 4
    data1 = data[i * 6 + 1]
    data2 = data[i* 6 + 4]
    result90, result100 = processingData(data1)
    mtx90[row, col] = result90
    mtx100[row, col] = result100
    result90, result100 = processingData(data2)
    mtx90R[row, col] = result90
    mtx100R[row, col] = result100
np.save("Active_Learning/feature90AL.npy", mtx90)
np.save("Active_Learning/feature100AL.npy", mtx100)
np.save("Active_Learning/feature90RAND.npy", mtx90R)
np.save("Active_Learning/feature100RAND.npy", mtx100R)

    
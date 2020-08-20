import numpy as np
import copy
import pandas as pd
from fancyimpute import SoftImpute
import os, sys
import matplotlib.pyplot as plt
from scipy.stats import entropy
import logging


def Generate_binary_result(prob):
    rand = np.random.uniform()
    if rand < prob:
        return True
    else: return False

def Preparing_Labels(labelnums):
    labels = []
    for i in range(0, labelnums):
        labels.append(i)
    return labels


def Generating_Truth(num, uniqueness, responsiveness, numtypes):
    labels = list(range(numtypes))
    table = []
    unique_num = int(num * uniqueness)
    for i in range(unique_num):
        row = []
        rand = np.random.randint(len(labels))
        row.append(labels[rand])
        table.append(row)
    for i in range(unique_num):
        rest_pool = labels.copy()
        rest_pool.remove(table[i][0])
        for j in range(unique_num - 1):
            if Generate_binary_result(responsiveness):
                table[i].append(rest_pool[np.random.randint(len(rest_pool))])
            else:
                table[i].append(table[i][0])
    for j in range(num - unique_num):
        rand = np.random.randint(unique_num)
        for i in range(unique_num):
            table[i].append(table[i][rand])
    for i in range(num - unique_num):
        table.append(table[np.random.randint(unique_num)])
    table = np.array(table)
    #table = cp.array(table)
    filename = str(uniqueness) + 'test_data.csv'
    # pd_data = pd.DataFrame(table)
    # pd_data.to_csv('test_data.csv',index=False,header=False)
    return table


def Initialize_Data(ground_truth, initial_num):
    currdata = []
    for i in range(len(ground_truth)):
        row = []
        for j in range(len(ground_truth[i])):
            row.append(-1)
        currdata.append(row)

    for i in range(initial_num):
        rand1 = np.random.randint(len(ground_truth))
        rand2 = np.random.randint(len(ground_truth[0]))
        while currdata[rand1][rand2] != -1 or ground_truth[rand1][rand2] == -1:
            rand1 = np.random.randint(len(ground_truth))
            rand2 = np.random.randint(len(ground_truth[0]))
        currdata[rand1][rand2] = ground_truth[rand1][rand2]
    print(currdata[0][0] == ground_truth[0][0])
    return currdata

def CountKnownNum(currdata):
    count = 0
    for row in currdata:
        for e in row:
            if e != -1:
                count+=1
    return count

def zeros_like(m):
    mtx = []
    for i in range(len(m)):
        row = []
        for j in range(len(m[i])):
            row.append(0)
        mtx.append(row)
    return mtx


def AddNoise(data, percent, phenotype):
    added = []
    noise_num = int(len(data) * len(data[0]) * percent)
    i = 0
    while i < noise_num:
        row = np.random.randint(0, len(data))
        col = np.random.randint(0, len(data[0]))
        while (row, col) in added:
             row = np.random.randint(0, len(data))
        col = np.random.randint(0, len(data[0]))
        noise_index = np.random.randint(0, phenotype)
        while data[row][col] == noise_index:
            noise_index = np.random.randint(0, phenotype)
        data[row][col] = noise_index
        added.append((row, col))
        i += 1
    return data

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def main(uniqueness, responsiveness, noise, obs, phenotype):
    ground_truth = Generating_Truth(50, uniqueness, responsiveness, phenotype)
    ground_truth = AddNoise(ground_truth, noise, phenotype)
    current_mtx = Initialize_Data(ground_truth, int(obs*2500))
    total = len(current_mtx) * len(current_mtx[0])
    affl_matrix = []
    for k in range(phenotype):
        mtx = zeros_like(current_mtx)
        affl_matrix.append(mtx)
    for i in range(len(current_mtx)):
        for j in range(len(current_mtx[i])):
            if current_mtx[i][j] != -1:
                idx = int(current_mtx[i][j])
                for k in range(phenotype):
                    if k == idx:
                        affl_matrix[k][i][j] = 1
            else: 
                for k in range(phenotype):
                    affl_matrix[k][i][j] = 'NaN'
    prediction = []
    with HiddenPrints():
        for k in range(len(mapping)):
            # solver = SoftImpute(J=2, maxit=1000).fit(affl_matrix[k])
            # affl_matrix[k] = SoftImpute(max_iters=500).fit_transform(affl_matrix[k])
            affl_matrix[k], _, _, _ = SoftImpute(max_iters=100, init_fill_method="half", verbose=False).fit_transform(affl_matrix[k])

    for i in range(len(current_mtx)):
        for j in range(len(current_mtx[i])):
            if current_mtx[i][j] == -1 and ground_truth[i, j] != -1:
                temp = []
                for k in range(len(mapping)):
                    temp.append(affl_matrix[k][i][j])
                label = mapping[np.argmax(temp)]
                prediction.append([i, j, label])

    affl_matrix.clear()
    masks.clear()
    mistake = 0
    for p in prediction:
        i = p[0]
        j = p[1]
        label = p[2]
        if ground_truth[i][j] != label:
            mistake += 1
    pred_accuracy = 1-mistake / len(prediction)


    filename = 'si_training_data.csv'
    result = []
    result.append([uniqueness, responsiveness, noise, obs, phenotype, pred_accuracy])
    pd_data.to_csv(filename,index=False,header=False,mode='a')



for p in [2, 4, 8, 12, 16, 24, 32, 48, 64]:
    for n in np.arange(0, 0.3, 0.05):
        for u in [0.2, 0.4, 0.6, 0.8]:
            for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
                for obs in np.arange(0.1, 1, 0.1):
                    main(u, r, n, obs, p)





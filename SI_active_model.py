import numpy as np
import copy
import pandas as pd
from soft_impute_test import SoftImpute
from soft_impute_test import SoftImpute
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

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] != -1:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T


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

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def Make_Selections_score2(num, predictions):
    batch = []
    for prediction in predictions:
        if len(batch) < num:
            batch.append(prediction)  #prediction format (row, col, label, score)
        else:
            max_score = batch[0][3]
            index = 0
            for i in range(len(batch)):
                if batch[i][3] > max_score:
                    max_score = batch[i][3]
                    index = i

            if prediction[3] < max_score:
                batch.remove(batch[index])
                batch.append(prediction)
    return batch

def Make_Selections_score(num, predictions):
    batch = []
    for prediction in predictions:
        if len(batch) < num:
            batch.append(prediction)  #prediction format (row, col, label, score)
        else:
            min_score = batch[0][3]
            index = 0
            for i in range(len(batch)):
                if batch[i][3] < min_score:
                    min_score = batch[i][3]
                    index = i
            if prediction[3] > min_score:
                batch.remove(batch[index])
                batch.append(prediction)
    return batch

def make_selections_random(num, predictions):
    batch = []
    batch_index = []
    for i in range(num):
        rand = np.random.randint(len(predictions))
        while rand in batch_index:
            rand = np.random.randint(len(predictions))
        batch_index.append(rand)
    for index in batch_index:
        batch.append(predictions[index])
    return batch

def main(uniqueness, responsiveness, noise, phenotype):
    ground_truth = Generating_Truth(100, uniqueness, responsiveness, phenotype)
    ground_truth = AddNoise(ground_truth, noise, phenotype)
    current_mtx = Initialize_Data(ground_truth, 100)
    knownnum = CountKnownNum(current_mtx)
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
    pred_accuracy =[]
    batch_size = 100
    accuracy = []
    while knownnum < total:
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
            for k in range(phenotype):
                affl_matrix[k] = SoftImpute(max_iters=500, init_fill_method="half", verbose=False).fit_transform(affl_matrix[k])

        query = 'margin'
        for i in range(len(current_mtx)):
            for j in range(len(current_mtx[i])):
                if current_mtx[i][j] == -1 and ground_truth[i, j] != -1:
                    temp = []
                    for k in range(phenotype):
                        temp.append(affl_matrix[k][i][j])
                    label = np.argmax(temp)
                    temp.sort(reverse=True)
                    if query == 'margin': 
                        if len(temp) == 1:
                            score = 0
                        else: score = temp[0] - temp[1]
                        prediction.append([i, j, label, score])
                    elif query == 'least_confidence':
                        score = temp[0]
                        prediction.append([i, j, label, score])
                    elif query == 'entropy':
                        temp2 = []
                        for value in temp:
                            temp2.append(value)
                        temp2 /= np.sum(temp2)
                        score = entropy(temp2, base=2)
                        prediction.append([i, j, label, score])
                    else: raise ValueError('unknown')
        size = min(batch_size, total - knownnum)
        if query == 'margin': 
            batch = Make_Selections_score2(size, prediction)
        affl_matrix.clear()
        mistake = 0
        for p in prediction:
            i = p[0]
            j = p[1]
            label = p[2]
            if ground_truth[i][j] != label:
                mistake += 1
        accuracy.append((total - mistake)/total)
        pred_accuracy.append(1-mistake / len(prediction))
        for item in batch:
            i = item[0]
            j = item[1]
            current_mtx[i][j] = ground_truth[i][j]
        knownnum = CountKnownNum(current_mtx)
    print(0)


np.random.seed(0)
main(0.6, 0.8, 0.1, 30)
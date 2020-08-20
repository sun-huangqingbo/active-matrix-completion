import numpy as np
import copy
import pandas as pd
from fancyimpute import SoftImpute, NuclearNormMinimization
# from soft_impute import SoftImpute
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

def main(uniqueness, responsiveness, query):
    phenotype = 15
    ground_truth = np.load("compound-target-datasets\experimental-space-v11-modified.npy")
    a = np.sum(np.where(ground_truth!=-1, 1, 0))
    percentages = []
    for k in range(1, 5):
        percentages.append(np.sum(np.where(ground_truth==k, 1, 0))/a)
    # plt.bar(range(1, 5), percentages)
    # plt.show()
    u, s, v = np.linalg.svd(ground_truth)
    rank = np.sum(s > 0.01*max(s))
    current_mtx = Initialize_Data(ground_truth, 150)
    total = len(current_mtx) * len(current_mtx[0])
    total0 = CountKnownNum(ground_truth)
    affl_matrix = []
    masks = []
    knownnum = CountKnownNum(current_mtx)
    accuracy = []
    pred_accuracy = []
    batch_size = 150#int(total0*0.01)
    while knownnum < total0:
        current_phenotypes = np.unique(current_mtx)
        del_idx = []
        for i in range(len(current_phenotypes)):
            if current_phenotypes[i] < 0:
                del_idx.append(i)
        current_phenotypes = np.delete(current_phenotypes, del_idx)
        mapping = {}
        mapping_inv = {}
        for i in range(len(current_phenotypes)):
            if current_phenotypes[i] > 0:
                mapping[i] = current_phenotypes[i]
                mapping_inv[current_phenotypes[i]] = i
        for k in range(len(mapping)):
            mtx = np.zeros_like(current_mtx).astype('f')
            affl_matrix.append(mtx)
        for i in range(len(current_mtx)):
            for j in range(len(current_mtx[i])):
                if current_mtx[i][j] != -1:
                    idx = mapping_inv[current_mtx[i][j]]
                    for k in range(len(mapping)):
                        if k == idx:
                            affl_matrix[k][i][j] = 1
                else: 
                    for k in range(len(mapping)):
                        affl_matrix[k][i][j] = "NaN"

        prediction = []
        with HiddenPrints():
            for k in range(len(mapping)):
                # solver = SoftImpute(J=2, maxit=1000).fit(affl_matrix[k])
                # affl_matrix[k] = SoftImpute(max_iters=500).fit_transform(affl_matrix[k])
                affl_matrix[k] = SoftImpute(max_iters=200, init_fill_method="half", verbose=False).fit_transform(affl_matrix[k])

        for i in range(len(current_mtx)):
            for j in range(len(current_mtx[i])):
                if current_mtx[i][j] == -1 and ground_truth[i, j] != -1:
                    temp = []
                    for k in range(len(mapping)):
                        temp.append(affl_matrix[k][i][j])
                    label = mapping[np.argmax(temp)]
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
        size = min(batch_size, total0 - knownnum)
        # if query == 'margin': 
        #     batch = Make_Selections_score2(size, prediction)
        # elif query == 'least_confidence':
        #     batch = Make_Selections_score2(1000, prediction)
        # elif query == 'entropy':
        #     batch = Make_Selections_score(1000, prediction)
        # else: raise ValueError('unknown')
        batch = make_selections_random(size, prediction)
        affl_matrix.clear()
        masks.clear()
        mistake = 0
        for p in prediction:
            i = p[0]
            j = p[1]
            label = p[2]
            if ground_truth[i][j] != label:
                mistake += 1
        accuracy.append((total0 - mistake)/total0)
        pred_accuracy.append(1-mistake / len(prediction))
        for item in batch:
            i = item[0]
            j = item[1]
            current_mtx[i][j] = ground_truth[i][j]
        knownnum = CountKnownNum(current_mtx)

    filename = 'softimpute_v11_random.csv'
    result = []
    result.append(accuracy)
    result.append(pred_accuracy)
    pd_data = pd.DataFrame(result)
    pd_data.to_csv(filename,index=False,header=False)
main(0.4, 0.2, "margin")




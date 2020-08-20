import numpy as np
import copy
import pandas as pd
# from soft_impute import SoftImpute
from soft_impute_test import SoftImpute
import os, sys
import matplotlib.pyplot as plt
from doc6b import *
import random

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
        while currdata[rand1][rand2] != -1:
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


def main():
    data = np.array(Readin_xls('elife-10047-supp4-v2.xls'))
    init_idx = np.random.choice(range(4*len(data)), int(0.04*len(data)), replace=False)
    measured_samples = []
    new_added = []
    measured_idx = {}
    init_instances = []
    for i in init_idx:
        row = int(i / 92)
        col = i % 92
        init_instances.append((row, col))
        if row > 46:
            row = row - 47
        if col > 45:
            col = col - 46
        new_added.append((row, col))

    for sample in data:
        row = sample[1]-1
        col = sample[0]-1
        if (row, col) in new_added and  (row, col) not in measured_idx:
            measured_samples.append(sample)
    for i in new_added:
        row = i[0]
        col = i[1]
        if (row, col) in measured_idx: 
            measured_idx[(row, col)] += 1
        else: measured_idx[(row, col)] = 1

   
    measured_samples = np.array(measured_samples)
    clustering = AgglomerativeClustering(distance_threshold=10, n_clusters=None).fit(measured_samples[:, 4:])
    measured_y = clustering.labels_

    knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance').fit(measured_samples[:, 4:], measured_y)
    ground_truth = Construct_Groundtruth(data, 47, 46, knn)
    current_mtx = -np.ones((94, 92))
    for i in init_instances:
        current_mtx[i[0], i[1]] = ground_truth[i[0], i[1]]
    total = len(current_mtx) * len(current_mtx[0])
    affl_matrix = []
    knownnum = CountKnownNum(current_mtx)
    accuracy = []
    vectors = []

    once = 0
    twice = 0
    thriple = 0
    quartic = 0
    for key in measured_idx:
        if measured_idx[key] == 1:
            once += 1
        elif measured_idx[key] == 2:
            twice += 1
        elif measured_idx[key] == 3:
            thriple += 1
        elif measured_idx[key] == 4:
            quartic += 1
    zeros = total/4 - once - twice - thriple - quartic
    vector = (zeros*4/total, once*4/total, twice*4/total, thriple*4/total, quartic*4/total)
    vectors.append(vector)

    while knownnum < total:
        phenotype = len(np.unique(measured_y))
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
                affl_matrix[k] = SoftImpute(max_iters=150, init_fill_method="half").fit_transform(affl_matrix[k])
        for i in range(len(current_mtx)):
            for j in range(len(current_mtx[i])):
                if current_mtx[i][j] == -1:
                    temp = []
                    for k in range(phenotype):
                        temp.append(affl_matrix[k][i][j])
                    label = np.argmax(temp)
                    temp.sort(reverse=True)
                    score = temp[0] - temp[1]
                    prediction.append([i, j, label, score])
        affl_matrix.clear()
        mistake = 0
        for p in prediction:
            i = p[0]
            j = p[1]
            label = p[2]
            if ground_truth[i][j] != label:
                mistake += 1
        accuracy.append((10000 - mistake)/10000)
        size = min(int(0.01*total), total-knownnum)
        # batch = make_selections_random(size, prediction)
        batch = Make_Selections_score2(size, prediction)
        new_added = []
        for i in range(len(batch)):
            current_mtx[batch[i][0]][batch[i][1]] = ground_truth[batch[i][0]][batch[i][1]]
            if batch[i][0] > 46:
                row = batch[i][0] - 47
            else: row = batch[i][0]
            if batch[i][1] > 45:
                col = batch[i][1] - 46
            else: col = batch[i][1]
            new_added.append((row, col))
        for sample in data:
            row = sample[1]-1
            col = sample[0]-1
            if (row, col) in new_added and (row, col) not in measured_idx:
                measured_samples = np.vstack((measured_samples, sample))
        for i in new_added:
            row = i[0]
            col = i[1]
            if (row, col) in measured_idx: 
                measured_idx[(row, col)] += 1
            else: measured_idx[(row, col)] = 1
        knownnum = CountKnownNum(current_mtx)
        once = 0
        twice = 0
        thriple = 0
        quartic = 0
        for key in measured_idx:
            if measured_idx[key] == 1:
                once += 1
            elif measured_idx[key] == 2:
                twice += 1
            elif measured_idx[key] == 3:
                thriple += 1
            elif measured_idx[key] == 4:
                quartic += 1
        zeros = total/4 - once - twice - thriple - quartic
        vector = (zeros*4/total, once*4/total, twice*4/total, thriple*4/total, quartic*4/total)
        vectors.append(vector)

        clustering = AgglomerativeClustering(distance_threshold=10, n_clusters=None).fit(measured_samples[:, 4:])
        #clustering = AgglomerativeClustering(n_clusters=60).fit(measured_samples[:, 4:])
        measured_y = clustering.labels_
        knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance').fit(measured_samples[:, 4:], measured_y)
        ground_truth = Construct_Groundtruth(data, 47, 46, knn)
        current_mtx = Update_currdata(current_mtx, ground_truth)

    filename = 'SoftImpute_AL_image_seed_10.csv'
    accuracy = np.array(accuracy)
    pd_data = pd.DataFrame(accuracy)
    pd_data.to_csv(filename,index=False,header=False)
    pd_vec = pd.DataFrame(vectors)
    pd_vec.to_csv('soft_impute_vec_seed_10.csv',index=False,header=False)
np.random.seed(10)
random.seed(10)
main()
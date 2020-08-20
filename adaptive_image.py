import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree
import matplotlib.pyplot as plt
from sklearn import datasets
import os
import math
import multiprocessing
import csv
import scipy
import random, copy 
from collections import Counter
from soft_impute_test import SoftImpute
from doc6b import *





def Read_in_csv(filename):
    csv_file=open(filename)     
    csv_reader_lines = csv.reader(csv_file)    
    date_PyList=[]  
    for one_line in csv_reader_lines:  
        date_PyList.append(one_line)    
    date_ndarray = np.array(date_PyList)
    print(date_ndarray)

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


def Preparing_Labels(labelnums):
    labels = []
    for i in range(0, labelnums):
        labels.append(i)
    return labels

def Generate_binary_result(prob):
    rand = np.random.uniform()
    if rand < prob:
        return True
    else: return False

def CountKnownNum(currdata):
    count = 0
    for row in currdata:
        for e in row:
            if e != -1:
                count+=1
    return count

def Make_Selections_score(num, predictions):
    prediction_ = copy.deepcopy(predictions)
    random.shuffle(prediction_)
    sorted_list = sorted(prediction_, key=lambda x : x[3], reverse = True)
    batch = sorted_list[: int(num)]
    return batch


def Make_Selections_score2(num, predictions):
    prediction_ = copy.deepcopy(predictions)
    random.shuffle(prediction_)
    sorted_list = sorted(prediction_, key=lambda x : x[3], reverse = False)
    batch = sorted_list[: int(num)]
    return batch





def Generating_Truth(num, uniqueness, responsiveness, numtypes):
    labels = Preparing_Labels(numtypes)
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

    return table


# def Initialize_Data(ground_truth, num):
#     currdata = []
#     for i in range(0, len(ground_truth) * len(ground_truth[0])):
#         currdata.append(-1)
#     currdata = np.array(currdata).reshape(len(ground_truth), len(ground_truth[0])) 
#     for i in range(len(ground_truth)):
#         label = ground_truth[i][i]
#         currdata[i][i] = label
#     return currdata

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
    return np.array(currdata)

    
        
def Find_remain_col(row, col, threshold, currdata):
    result = {}
    cols = []
    for i in range(len(currdata[0])):
        cols.append(i)
    cols.remove(col)
    for col2 in cols:
        if currdata[row][col2] != -1:
            score = Count_Col_Score(col, col2, currdata)
            result[col2] = score

    batchsize = max(threshold, 1)
    maxcols = []
    for key in result:
        if len(maxcols) < batchsize - 1:
            maxcols.append(key)
        elif len(maxcols) == batchsize - 1:
            maxcols.append(key)
            mincol = maxcols[0]
            for col_ in maxcols:
                if result[col_] < result[mincol]:
                    mincol = col_
        else:
            if result[key] > result[mincol]:
                maxcols.remove(mincol)
                maxcols.append(key)
                mincol = maxcols[0]
                for col_ in maxcols:
                    if result[col_] < result[mincol]:
                        mincol = col_

    return maxcols




def Predict_One_Col(col, currdata, step, panelty):
    n_conflicts = 0
    predictions = []
    #index is consistent with currdata
    unknown_rows = []


    for row in range(len(currdata)):
        if currdata[row][col] == -1:
            unknown_rows.append(row)

    while True:
        if len(unknown_rows) == 0 or n_conflicts > len(currdata):
            break
        predicted = []
        community = Find_Cols_With_n_Conflicts(n_conflicts + step, n_conflicts, currdata, col)
        if len(community) != 0:
            for urow in unknown_rows:
                prediction = {}
                count = {}
                for col2 in community:
                    if currdata[urow][col2] != -1:
                        label = currdata[urow][col2]
                        if label in prediction:
                            prediction[label] += Count_Col_Score(col, col2, panelty, currdata)
                            count[label] += 1
                        else:
                            prediction[label] = Count_Col_Score(col, col2, panelty, currdata)
                            count[label] = 1
                
                for label in prediction:
                    prediction[label] = prediction[label]/count[label]
                

                if len(prediction) != 0:
                    predictions.append((urow, col, prediction))
                    predicted.append(urow)
            n_conflicts += step
            for row in predicted:
                unknown_rows.remove(row)
        else:
            n_conflicts += step
    for urow in unknown_rows:
        predictions.append((urow, col, -2))
    return predictions

def Count_Row_Score(row, currdata):
    count = 0
    for col in range(len(currdata[0])):
        if currdata[row][col] != -1:
            count+=1
    return count


def Count_Col_Score(col, col2, panelty, currdata):
    score = 0
    same = 0
    conflict = 1
    for row in currdata:
        if row[col] != -1 and row[col2] != -1 and row[col] == row[col2]:
            score += 1
            same += 1
        if row[col] != -1 and row[col2] != -1 and row[col] != row[col2]:
            score -= panelty
            conflict+=1
    return score

def Find_Cols_With_n_Conflicts(upperbound, lowerbound, currdata, col):
    #row is index of rows in currdata
    result = []
    cols = []
    for i in range(len(currdata[0])):
        cols.append(i)
    cols.remove(col)
    for col2 in cols:
        conflict = 0
        for row in range(len(currdata)):
            if currdata[row][col] != -1 and currdata[row][col2] != -1 and currdata[row][col] != currdata[row][col2]:
                conflict += 1
        if conflict <  upperbound and conflict >= lowerbound:
            result.append(col2)

    return result



def Split_Parameters(pl):
    col = pl[0]
    currdata = pl[1]
    step = pl[2]
    panelty = pl[3]
    return Predict_One_Col(col, currdata, step, panelty)


def Simulate_Once(currdata, step, panelty):
    prediction_from_col = []
    prediction_from_row = []
    predictions = {}
    parameters = []
    for col in range(len(currdata[0])):
        parameters.append((col, currdata, step, panelty))
    

    core_num = min(26, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(core_num)
    predictions_col = pool.map(Split_Parameters, parameters)
    pool.close()
    pool.join()
        #p: row, col, dict

    for l in predictions_col:
        for p in l:
            prediction_from_col.append(p)
    for p in prediction_from_col:
        predictions[(p[0], p[1])] = p[2]
    
    
    currdataT = currdata.T
    parameters.clear()
    
    for col_ in range(len(currdataT[0])):
        parameters.append((col_, currdataT, step, panelty))

    core_num = min(26, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(core_num)
    predictions_row = pool.map(Split_Parameters, parameters)
    pool.close()
    pool.join()

    for l in predictions_row:
        for p in l:
            prediction_from_row.append(p)

    for p in prediction_from_row:
        if (p[1], p[0]) in predictions and predictions[(p[1], p[0])] != -2 and p[2] != -2:
            for key in p[2]:
                if key in predictions[(p[1], p[0])]:
                    predictions[(p[1], p[0])][key] = p[2][key] + predictions[(p[1], p[0])][key]
                    #predictions[(p[1], p[0])][key] /= 2
                else:
                    predictions[(p[1], p[0])][key] = p[2][key]
        elif (p[1], p[0]) in predictions and predictions[(p[1], p[0])] != -2  and p[2] == -2:
            pass
        elif (p[1], p[0]) in predictions and predictions[(p[1], p[0])] == -2:
            if p[2] != -2:
                for key in p[2]:
                    predictions[(p[1], p[0])] = {}
                    predictions[(p[1], p[0])][key] = p[2][key]
        else: 
            predictions[(p[1], p[0])][key] = p[2]

    return predictions


def Make_Selections_random(num, predictions):
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





def Calculate_Resp(currdata):
    resps = []
    resps2 = []
    for row in currdata:
        resp = 0
        row_label = {}
        for e in row:
            if e != -1 and e not in row_label:
                row_label[e] = 1
            elif e != -1 and e in row_label:
                row_label[e] += 1
        max_label = -2
        maxscore = -1000000
        for key in row_label:
            if row_label[key] > maxscore:
                maxscore = row_label[key]
                max_label = key
        resp = maxscore
        total = maxscore
        for key in row_label:
            if key != max_label:
                total += row_label[key]
        resps.append((max_label, resp))
        resps2.append(maxscore/total)
    resps2 = np.array(resps2)
    resp_average = resps2.mean()
    return 1 - resp_average

def estimate_resp(a):
    resp = []
    for row in a:
        temp = []
        for e in row:
            if e != -1:
                temp.append(e)
        if len(temp) == 0:
            continue
        counts = np.bincount(temp)
        resp.append(1 - np.max(counts)/len(temp))
    resp = np.array(resp)
    return resp.mean()


def estimate_uniq(a):
    uniq = 0
    for i in range(100):
        row = a[i]
        temp = []
        for e in row:
            if e != -1:
                temp.append(e)
        committee = Find_Cols_With_n_Conflicts(int(0.1*len(temp)), 0, a.T, i)
        if len(committee) == 0:
            uniq+=1
    a = a.T
    for i in range(100):
        row = a[i]
        temp = []
        for e in row:
            if e != -1:
                temp.append(e)
        committee = Find_Cols_With_n_Conflicts(int(0.1*len(temp)), 0, a.T, i)
        if len(committee) == 0:
            uniq+=1
    return uniq/200.0


def Calculate_entropy(prediction):
    entropy = 0
    for key in prediction:
        p = prediction[key]
        entropy -= p * scipy.log2(p)
    return entropy

def row_common_label(mtx):
    label_set = []
    for i in range(len(mtx)):
        temp = {}
        for j in range(len(mtx[i])):
            label = mtx[i][j]
            if label != -1 and label not in temp:
                temp[label] = 1
            elif label != -1 and label in temp:
                temp[label] += 1
        temp_sorted = sorted(temp.items(), key=lambda x:x[1], reverse = True)
        if len(temp_sorted) > 0:
            label_set.append([temp_sorted[0][0], temp_sorted[0][1]])
        else:
            label_set.append([-2, -2])
    return label_set

def global_common_label(mtx):
    temp = {}
    for i in range(len(mtx)):
        for j in range(len(mtx[i])):
            label = mtx[i][j]
            if label != -1 and label not in temp:
                temp[label] = 1
            elif label != -1 and label in temp:
                temp[label] += 1
        temp_sorted = sorted(temp.items(), key=lambda x:x[1], reverse = True)
    return temp_sorted[0][0]



def ibc(currdata, ground_truth10):
    total = len(currdata) * len(currdata[0])
    knownnum = CountKnownNum(currdata)
    P = 1
    batch_size = 100
    step = 1
    panelty = 1 * (knownnum/total)
    prediction_temp = Simulate_Once(currdata, step, panelty)
    row_common_label_ = row_common_label(currdata)
    col_common_label_ = row_common_label(currdata.T)
    global_common_label_ = global_common_label(currdata)
    predictions = []
    count = 0
    correct = 0
    mistake = []
    corrects = []
    for key in prediction_temp:
        row = key[0]
        col = key[1]
        if ground_truth10[row, col] != -1:
            count += 1
            dictp = prediction_temp[key]
            maxscore = -10000
            maxlabel = ''
            if dictp != -2 :
                for label in dictp:
                        if dictp[label] > maxscore:
                            maxlabel = label
                            maxscore = dictp[label]
            else:
                if row_common_label_[row][0] != -2 and col_common_label_[col][0] != -2:
                    row_score = row_common_label_[row][1]
                    col_score = col_common_label_[col][1]
                    if col_score > row_score:
                        maxlabel = row_common_label_[row][0]
                    else:
                        maxlabel = col_common_label_[col][0]
                    maxscore = -panelty * 300
                elif row_common_label_[row][0] != -2 and col_common_label_[col][0] == -2:
                    maxlabel = row_common_label_[row][0]
                    maxscore = -panelty * 300
                elif row_common_label_[row][0] == -2 and col_common_label_[col][0] != -2:
                    maxlabel = col_common_label_[col][0]
                    maxscore = -panelty * 300
                else:
                    maxlabel = global_common_label_
                    maxscore = -panelty * 300
            # if args.adjustment:
            #     if maxscore < curr_resp[row][1]:
            #         maxlabel = curr_resp[row][0]
            
            if ground_truth10[row][col] == maxlabel:
                    correct+=1
                    corrects.append(maxscore)
            else:
                mistake.append(maxscore)
            #TF_ratio = Calculate_TrueFalsth_Ratio(prediction_temp[key], maxlabel)
            if dictp != -2:
                bias = 1 - min(dictp.values())
                for key in dictp:
                    dictp[key] += bias
                k = 1 / sum(dictp.values())
                for key in dictp:
                    dictp[key] *= k
                uncertain_score = Calculate_entropy(dictp)
            else: uncertain_score = 1
            predictions.append([row, col, maxlabel, maxscore, uncertain_score])
    knownnum = CountKnownNum(currdata)

    accu = 1 - (count - correct)/total

    pred_accuracy = correct / count
    size = min(batch_size, total - knownnum) 
    selections1 = Make_Selections_score2(size*0.5, predictions)
    for p in selections1:
        predictions.remove(p)
    for p in predictions:
        p[3] = p[4]
    selections2 = Make_Selections_score(size - int(size*0.5), predictions)
    selections = []
    for item in selections2:
        selections.append(item)
    for item in selections1:
        selections.append(item)
    return accu, selections

def si(current_mtx, ground_truth):
    knownnum = CountKnownNum(current_mtx)
    total = len(current_mtx) * len(current_mtx[0])
    affl_matrix = []
    current_phenotypes = np.unique(current_mtx)
    del_idx = []
    for i in range(len(current_phenotypes)):
        if current_phenotypes[i] < 0:
            del_idx.append(i)
    current_phenotypes = np.delete(current_phenotypes, del_idx)
    mapping = {}
    mapping_inv = {}
    for i in range(len(current_phenotypes)):
        if current_phenotypes[i] >= 0:
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
                    affl_matrix[k][i][j] = 'NaN'
    prediction = []
    batch_size = 100
    

    prediction = []

    for k in range(len(mapping)):
        affl_matrix[k] = SoftImpute(max_iters=100, init_fill_method="half", verbose=False).fit_transform(affl_matrix[k])

    for i in range(len(current_mtx)):
        for j in range(len(current_mtx[i])):
            if current_mtx[i][j] == -1 and ground_truth[i, j] != -1:
                temp = []
                for k in range(len(mapping)):
                    temp.append(affl_matrix[k][i][j])
                label = mapping[np.argmax(temp)]
                temp.sort(reverse=True)
    
                if len(temp) == 1:
                    score = 0
                else: score = temp[0] - temp[1]
                prediction.append([i, j, label, score])
                
    size = min(batch_size, total - knownnum) 
    batch = Make_Selections_score2(size, prediction)
    affl_matrix.clear()
    mistake = 0
    for p in prediction:
        i = p[0]
        j = p[1]
        label = p[2]
        if ground_truth[i][j] != label:
            mistake += 1
    return 1-mistake / total, batch


def main(u, r, noise, tree):
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
    known = CountKnownNum(current_mtx)
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
    while known < total:
        obs = known/total
        est_resp = estimate_resp(current_mtx)
        phenotypes = np.unique(current_mtx)
        p_num = len(phenotypes)-1
        predict = tree.predict([[est_resp, obs]])
        if predict[0] == 1:
            a, batch = ibc(current_mtx, ground_truth)
        else:
            a, batch = si(current_mtx, ground_truth)
        accuracy.append(a)
        print(a)
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

    filename = 'adaptive_image_results_seed10.csv'
    result = []
    result.append(accuracy)
    pd_data = pd.DataFrame(result)
    pd_data.to_csv(filename,index=False,header=False)
    pd_vec = pd.DataFrame(vectors)
    pd_vec.to_csv('adaptive_vec.csv',index=False,header=False)
# ------------------------------------------------------------


if __name__ == "__main__":
    np.random.seed(10)
    random.seed(10)
    file = "IBC_active_training_data_v2.csv"
    file2 = "SI_active_training_data.csv"
    data = np.genfromtxt(file, delimiter=',')
    data2 = np.genfromtxt(file2, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    X2 = data2[:, :-1] 
    y2 = data2[:, -1]
    poly = PolynomialFeatures(4)

    X = poly.fit_transform(X)
    X2 = poly.transform(X2)
    # print(poly.powers_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=0)
    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    reg2 = LinearRegression().fit(X_train2, y_train2)
    print(reg2.score(X_test2, y_test2))
    X_all = []
    y_all = []
    for p in np.arange(2, 48, 4):
        for n in np.arange(0, 0.3, 0.05):
            for u in np.arange(0, 1, 0.1):
                for r in np.arange(0, 1, 0.1):
                    for obs in np.arange(0, 1, 0.01):
                        temp = poly.transform([[u, r, n, obs, p]])
                        X_all.append([r, obs])
                        y1 = reg.predict(temp)
                        y2 = reg2.predict(temp)
                        if y1 >= y2:
                            y_all.append(1)
                        else: y_all.append(0)
    print(np.unique(y_all))
    X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, random_state=0)
    tree = DecisionTreeClassifier(random_state=0, max_depth=3, criterion="entropy").fit(X_all, y_all)
    print(tree.score(X_all_test, y_all_test))
    main(0, 0, 0, tree)




# fig = plt.figure(figsize=(250,200))
# dot_data = sklearn.tree.plot_tree(clf,  
#                                 feature_names=["uniqueness", "responsiveness", "noise", "observation", "phenotype number"],  
#                                 class_names=["SOFT IMPUTE-Based", "IBC"],
#                                 filled=True)

# # Draw graph
# fig.savefig("decistion_tree_active.png")

    
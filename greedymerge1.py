import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import random
def Generate_binary_result(prob):
    rand = np.random.uniform()
    if rand < prob:
        return True
    else: return False

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


def check_two_distributions(dist1, dist2, num):
    for i in range(num):   #100
        labels = []
        for key in dist1.keys():
            if dist1[key][i] != -1:
                labels.append(dist1[key][i])
        for key in dist2.keys():
            if dist2[key][i] != -1:
                labels.append(dist2[key][i])
        if len(np.unique(labels)) > 1:
            return False
    return True

def merge_two_distributions(dist1, dist2):
    new_dist = {}
    for key in dist1.keys():
        new_dist[key] = dist1[key]
    for key in dist2.keys():
        new_dist[key] = dist2[key]
    return new_dist

def initialize_distributions(current_mtx):
    distribution_set = []
    for i in range(len(current_mtx)):
        dist = {}
        dist[i] = copy.deepcopy(current_mtx[i])
        distribution_set.append(dist)
    return distribution_set

def get_label(distribution, key_, idx):
    label = []
    for key in distribution.keys():
        if key != key_:
            if distribution[key][idx] != -1:
                label.append(distribution[key][idx])
    label = np.unique(label)
    if len(label) > 1:
        raise ValueError('more than 1 labels in one distribution!')
    elif len(label) == 1:
        return label[0]
    else: return -2

def count_distribution_score(distribution): 
    count = 0
    for key in distribution.keys():
        vec = distribution[key]
        for i in range(len(vec)):
            if vec[i] != -1:
                count += 1
    return count

def impute_from_distribution(distribution_set):
    prediction = {}
    for distribution in distribution_set:
        score = count_distribution_score(distribution)
        for key in distribution.keys():
            vec = distribution[key]
            for i in range(len(vec)):
                if vec[i] == -1:
                    label = get_label(distribution, key, i)
                    prediction[(key, i)] = [label, score] 
    return prediction

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
            label_set.append(temp_sorted[0][0])
        else:
            label_set.append(0)
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

def Make_Selections_score2(num, prediction):
    batch = []
    for key in prediction.keys():
        if len(batch) < num:
            batch.append([key[0], key[1], prediction[key][0], prediction[key][1]])  #prediction format (row, col, label, score)
        else:
            max_score = batch[0][3]
            index = 0
            for i in range(len(batch)):
                if batch[i][3] > max_score:
                    max_score = batch[i][3]
                    index = i

            if prediction[key][1] < max_score:
                batch.remove(batch[index])
                batch.append([key[0], key[1], prediction[key][0], prediction[key][1]])
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


def main(uniqueness, responsiveness):
    phenotype = 32
    ground_truth = np.load("compound-target-datasets\experimental-space-v11.npy")
    # ground_truth = Generating_Truth(100, uniqueness, responsiveness, 32)
    # ground_truth = AddNoise(ground_truth, 0.1, 32)
    current_mtx = Initialize_Data(ground_truth, 500)
    total = len(current_mtx) * len(current_mtx[0])
    affl_matrix = []
    knownnum = CountKnownNum(current_mtx)
    accuracy = []
    pred_accuracy = []
    total0 = CountKnownNum(ground_truth)
    batch_size = 500
    

    while knownnum < total0:
        distribution_set = initialize_distributions(current_mtx)
        sig = False
        while not sig:
            is_looping = True
            for dist1 in distribution_set:
                for dist2 in distribution_set:
                    if dist1 != dist2:
                        if check_two_distributions(dist1, dist2, len(current_mtx[0])):
                            new = merge_two_distributions(dist1, dist2)
                            distribution_set.append(new)
                            distribution_set.remove(dist1)
                            distribution_set.remove(dist2)
                            is_looping = False
                            break
                if not is_looping:
                    break
            if is_looping:
                sig = True
        prediction_from_row = impute_from_distribution(distribution_set)
        prediction = {}
        for key in prediction_from_row.keys():
            prediction[key] = prediction_from_row[key]

        current_mtx = np.array(current_mtx)
        current_mtx_T = copy.deepcopy(current_mtx).T
        distribution_set = initialize_distributions(current_mtx_T)
        sig = False
        while not sig:
            is_looping = True
            for dist1 in distribution_set:
                for dist2 in distribution_set:
                    if dist1 != dist2:
                        if check_two_distributions(dist1, dist2, len(current_mtx_T[0])):
                            new = merge_two_distributions(dist1, dist2)
                            distribution_set.append(new)
                            distribution_set.remove(dist1)
                            distribution_set.remove(dist2)
                            is_looping = False
                            break
                if not is_looping:
                    break
            if is_looping:
                sig = True
        prediction_from_column = impute_from_distribution(distribution_set)
        row_common_label_ = row_common_label(current_mtx)
        global_common_label_ = global_common_label(current_mtx)
        for key in prediction_from_column.keys():
            row = key[1]
            col = key[0]
            if prediction[(row, col)][0] == -2:
                if prediction_from_column[key][0] != -2:
                    prediction[(row, col)] = prediction_from_column[key]
                elif row_common_label_[row] != -2:
                    prediction[(row, col)] = [row_common_label_[row], 0]
                    
                else: prediction[(row, col)] = [global_common_label_[row], 0]
            else:
                if prediction[(row, col)][0] == prediction_from_column[key][0]:
                    prediction[(row, col)][1] += prediction_from_column[key][1]
                else:
                    if prediction[(row, col)][1] < prediction_from_column[key][1]:
                        prediction[(row, col)] = prediction_from_column[key]
        mistake = 0
        count = 0
        removed = []
        for key in prediction.keys():
            row = key[0]
            col = key[1]
            if ground_truth[row][col] != -1:
                count+=1
                if prediction[key][0] != ground_truth[row][col]:
                    mistake += 1
            else: removed.append(key)
        for key in removed:
            del prediction[key]
        accuracy.append(1-mistake/total0)
        pred_accuracy.append(1-mistake/count)
        size = min(batch_size, total0 - knownnum)
        pred_list = []
        for key in prediction:
            pred_list.append([key[0], key[1], prediction[key][0], prediction[key][1]])
        batch = Make_Selections_score2(size, prediction)
        for item in batch:
            i = item[0]
            j = item[1]
            current_mtx[i][j] = ground_truth[i][j]
        knownnum = CountKnownNum(current_mtx)

    filename = 'GM_seed_10_CT_AL.csv'
    result = []
    result = []
    # result.append([uniqueness, responsiveness])
    result.append(accuracy)
    result.append(pred_accuracy)
    pd_data = pd.DataFrame(result)
    pd_data.to_csv(filename, index=False, header=False, mode="a")
np.random.seed(10)
random.seed(10)
main(1,1)


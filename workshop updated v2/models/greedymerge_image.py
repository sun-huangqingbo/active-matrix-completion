import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
from doc6b import *

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

def Make_Selections_score2(num, predictions):
    batch = []
    for prediction in predictions:
        if len(batch) < num:
            batch.append(prediction)  #prediction format (row, col, label, score)
        else:
            max_score = batch[0][1]
            index = 0
            for i in range(len(batch)):
                if batch[i][1] > max_score:
                    max_score = batch[i][1]
                    index = i

            if prediction[1] < max_score:
                batch.remove(batch[index])
                batch.append(prediction)
    return batch

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

def make_selections_random(num, predictions):
    keys = list(predictions.keys())
    batch = []
    batch_index = []
    for i in range(num):
        rand = np.random.randint(len(keys))
        while rand in batch_index:
            rand = np.random.randint(len(keys))
        batch_index.append(rand)
    for index in batch_index:
        batch.append(keys[index])
    return batch


def main(strategy):
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
        distribution_set = initialize_distributions(current_mtx)
        sig = False
        while not sig:
            is_looping = True
            for dist1 in distribution_set:
                for dist2 in distribution_set:
                    if dist1 != dist2:
                        if check_two_distributions(dist1, dist2, 92):
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
                        if check_two_distributions(dist1, dist2, 94):
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
                    prediction[(row, col)][1] = prediction_from_column[key][1] + prediction[(row, col)][1]
                else:
                    if prediction[(row, col)][1] < prediction_from_column[key][1]:
                        prediction[(row, col)] = prediction_from_column[key]
        mistake = 0
        for key in prediction.keys():
            row = key[0]
            col = key[1]
            if prediction[key][0] != ground_truth[row][col]:
                mistake += 1
        accuracy.append(1-mistake/total)
        size = min(int(0.01*total), total-knownnum)
        if strategy == 'active':
            batch = Make_Selections_score2(size, prediction)
        else:
            batch = make_selections_random(size, prediction)

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

    return accuracy
    # filename = 'GreedyMerge_' + 'AL_real_v3.csv'
    # accuracy = np.array(accuracy)
    # pd_data = pd.DataFrame(accuracy.T)
    # pd_data.to_csv(filename,index=False,header=False)
    # pd_vec = pd.DataFrame(vectors)
    # pd_vec.to_csv('GreedyMerge_vec.csv',index=False,header=False)




# main()
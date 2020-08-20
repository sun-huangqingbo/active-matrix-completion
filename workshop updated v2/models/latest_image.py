import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import multiprocessing
import csv
import scipy
import argparse
from doc6b import *


def Read_in_csv(filename):
    csv_file=open(filename)     
    csv_reader_lines = csv.reader(csv_file)    
    date_PyList=[]  
    for one_line in csv_reader_lines:  
        date_PyList.append(one_line)    
    date_ndarray = np.array(date_PyList)
    print(date_ndarray)

def AddNoise(data, percent):
    added = []
    noise_num = int(len(data) * len(data[0]) * percent)
    i = 0
    while i < noise_num:
        row = np.random.randint(0, 99)
        col = np.random.randint(0, 99)
        while (row, col) in added:
            row = np.random.randint(0, 99)
            col = np.random.randint(0, 99)
        noise_index = np.random.randint(0, 8)
        while data[row][col] == str(noise_index):
            noise_index = np.random.randint(0, 8)
        data[row][col] = str(noise_index)
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
    #table = cp.array(table)
    filename = str(uniqueness) + 'test_data.csv'
    pd_data = pd.DataFrame(table)
    pd_data.to_csv('test_data.csv',index=False,header=False)
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
        while currdata[rand1][rand2] != -1:
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
    for row in currdata:
        if row[col] != -1 and row[col2] != -1 and row[col] == row[col2]:
            score += 1
        if row[col] != -1 and row[col2] != -1 and row[col] != row[col2]:
            score -= panelty
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
    

    core_num = 6
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

    core_num = multiprocessing.cpu_count()
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



def Calculate_Resp(currdata, panelty):
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
                resp -= panelty * row_label[key]
                total += row_label[key]
        resps.append((max_label, resp))
        resps2.append(maxscore/total)
    resps2 = np.array(resps2)
    resp_average = resps2.mean()
    return resps, 1 - resp_average

def Calculate_entropy(prediction):
    entropy = 0
    for key in prediction:
        p = prediction[key]
        entropy -= p * scipy.log2(p)
    return entropy


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
            label_set.append(-2)
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

#-------------------------------------------------------------------------------------------------------------------------
#active learning
    while knownnum < total:
        step = 1
        panelty = 2.5 * (knownnum/total)
        prediction_temp = Simulate_Once(current_mtx, step, panelty)
        curr_resp, resp_average = Calculate_Resp(current_mtx, panelty)
        row_common_label_ = row_common_label(current_mtx)
        global_common_label_ = global_common_label(current_mtx)
        predictions = []
        count = len(prediction_temp)
        correct = 0
        mistake = []
        corrects = []
        for key in prediction_temp:
            row = key[0]
            col = key[1]
            dictp = prediction_temp[key]
            maxscore = -10000
            maxlabel = ''
            if dictp != -2 :
                for label in dictp:
                        if dictp[label] > maxscore:
                            maxlabel = label
                            maxscore = dictp[label]
            else:
                if row_common_label_[row] != -2:
                    maxlabel = curr_resp[row][0]
                    maxscore = -panelty * 100
                else:
                    maxlabel = global_common_label_
                    maxscore = -panelty * 100
            # if args.adjustment:
            #     if maxscore < curr_resp[row][1]:
            #         maxlabel = curr_resp[row][0]
            
            if ground_truth[row][col] == maxlabel:
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
        knownnum = CountKnownNum(current_mtx)

        accu = 1 - (count - correct)/total
        accuracy.append(accu)

        size = min(int(0.01*total), total - knownnum)
        if strategy == 'hybrid':
            selections = Make_Selections_score2(size*0.5, predictions)
            for p in selections:
                predictions.remove(p)
            for p in predictions:
                p[3] = p[4]
            selections2 = Make_Selections_score(size*0.5, predictions)
        elif strategy == 'entropy':
            selections = []
            for p in selections:
                predictions.remove(p)
            for p in predictions:
                p[3] = p[4]
            selections2 = Make_Selections_score(size, predictions)
        elif strategy == 'score':
            selections = Make_Selections_score2(size, predictions)
            selections2 = []
        elif strategy == 'random':
            selections = make_selections_random(size, predictions)
            selections2 = []
        else: raise ValueError('unknown query strategy.')

        batch = []
        for selection in selections:
            row = selection[0]
            col = selection[1]
            current_mtx[row][col] = ground_truth[row][col]
            batch.append(selection)
        for selection in selections2:
            row = selection[0]
            col = selection[1]
            current_mtx[row][col] = ground_truth[row][col]
            batch.append(selection)
        knownnum = CountKnownNum(current_mtx)


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

    # filename = 'ibc_AL_real_v2.csv'
    
    # #result.append(accusR.values())
    # accuracy = np.array(accuracy)
    # pd_data = pd.DataFrame(accuracy)
    # pd_data.to_csv(filename,index=False,header=False)
    # pd_vec = pd.DataFrame(vectors)
    # pd_vec.to_csv('IBC_vec.csv',index=False,header=False)



# if __name__ == "__main__":
#     main()


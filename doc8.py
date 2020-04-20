import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import multiprocessing
import csv
import scipy
from doc6b import *
import random

def Read_in_csv(filename):
    csv_file=open(filename)    #打开文件  
    csv_reader_lines = csv.reader(csv_file)    #用csv.reader读文件  
    date_PyList=[]  
    for one_line in csv_reader_lines:  
        date_PyList.append(one_line)    #逐行将读到的文件存入python的列表  
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

def Make_Selections_Higher_Score(num, predictions, batch):#Entropy
    random.shuffle(predictions)
    '''
    predictions = np.array(predictions)
    idex = np.lexsort([-1*predictions[:,4]])
    sorted_predictions = predictions[idex, :]
    '''
    predictions.sort(reverse=True, key=lambda x: x[4])

    batch_len = 0
    i = 0
    while batch_len < num:
        random.shuffle(predictions)
        predictions.sort(reverse=True, key=lambda x: x[4])
        row = int(predictions[i][0])
        col = int(predictions[i][1])
        if (row, col) not in batch:
            batch.append((row, col))
            batch_len += 1
        i = (i + 1 if i < (len(predictions)-1) else 0)


def Make_Selections_Lower_Score(num, predictions, batch):
    random.shuffle(predictions)
    '''
    predictions = np.array(predictions)
    idex = np.lexsort([predictions[:,3]])
    sorted_predictions = predictions[idex, :]
    '''
    predictions.sort(reverse=False, key=lambda x: x[3])
    batch_len = 0
    i = 0
    while batch_len < num:
        random.shuffle(predictions)
        predictions.sort(reverse=False, key=lambda x: x[3])
        row = int(predictions[i][0])
        col = int(predictions[i][1])
        if (row, col) not in batch:
            batch.append((row, col))
            batch_len += 1
        i = (i + 1 if i < (len(predictions)-1) else 0)






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

    table = np.array(table, dtype = 'int16')
    #table = cp.array(table)
    filename = str(uniqueness) + 'test_data.csv'

    pd_data = pd.DataFrame(table)
    pd_data.to_csv('test_data.csv',index=False,header=False)


    
    return table


def Initialize_Data(ground_truth, observed_percent):
    observednum = max(0, int(len(ground_truth[0]) * len(ground_truth) * observed_percent) - len(ground_truth[0]) - len(ground_truth))
    currdata = []

    currdata = np.ones((len(ground_truth), len(ground_truth[0])), 'int16')
    currdata *= -1

    for i in range(len(ground_truth)):
        label = ground_truth[i][i]
        currdata[i][i] = label

    for i in range(observednum):
        rand1 = np.random.randint(len(ground_truth))
        rand2 = np.random.randint(len(ground_truth[0]))
        while currdata[rand1][rand2] != -1:
            rand1 = np.random.randint(len(ground_truth))
            rand2 = np.random.randint(len(ground_truth[0]))
        currdata[rand1][rand2] = ground_truth[rand1][rand2]
    print(currdata[0][0] == ground_truth[0][0])
    return currdata


    
        
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
    

    core_num = min(multiprocessing.cpu_count(), 10)
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
    
    '''
    currdataT = currdata.T
    parameters.clear()
    
    for col_ in range(len(currdataT[0])):
        parameters.append((col_, currdataT, step, panelty))

    core_num = min(multiprocessing.cpu_count(), 10)
    pool = multiprocessing.Pool(core_num)
    predictions_row = pool.map(Split_Parameters, parameters)
    pool.close()
    pool.join()

    for l in predictions_row:
        for p in l:
            prediction_from_row.append(p)

    for p in prediction_from_row:
        if (p[1], p[0]) in predictions:
            for key in p[2]:
                if key in predictions[(p[1], p[0])]:
                    predictions[(p[1], p[0])][key] = p[2][key] + predictions[(p[1], p[0])][key]
                    #predictions[(p[1], p[0])][key] = max(p[2][key], predictions[(p[1], p[0])][key])
                    #predictions[(p[1], p[0])][key] = predictions[(p[1], p[0])][key]
                else:
                    predictions[(p[1], p[0])][key] = p[2][key]
        else:
            predictions[(p[1], p[0])][key] = p[2]
    '''
    return predictions


def Make_Selections_random(num, predictions, batch):
    batch_index = []
    for i in range(num):
        rand = np.random.randint(len(predictions))
        while rand in batch_index:
            rand = np.random.randint(len(predictions))
        batch_index.append(rand)

    for index in batch_index:
        batch.append(predictions[index])






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

        max_label = ''
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


def Trail_ActiveLearning(uniqueness, responsiveness, initial_size):
    #ground_truth_ = Generating_Truth(100, uniqueness, responsiveness, 150)
    #ground_truth10 = ground_truth_
    #ground_truth10 = AddNoise(ground_truth_, 0.1)
    #currdata = Initialize_Data(ground_truth10, initial_size)
    data = np.array(Readin_xls('E:\Active_Learning\elife\elife-10047-supp4-v2.xls'))
    clustering = AgglomerativeClustering(distance_threshold=5, n_clusters=None).fit(data[:, 4:])
    XY = np.hstack((data, clustering.labels_.reshape(-1,1)))

    measured_samples_ = {}
    measured_initial = []
   
    for sample in XY:
        row = sample[1]-1
        col = sample[0]-1
        #if row == col or (row == 92 and col == 91) or (row == 93 and col == 91):
        if row == 0 or col == 0:
        #measured_samples_.append((row, col))
            measured_initial.append(sample)
    measured_samples = np.array(measured_initial)

    current_labels = measured_samples[:,-1].copy().astype(np.int16)
    knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance').fit(measured_samples[:, 4:-1], current_labels)
    ground_truth10 = Update_Groundtruth(XY, 47, 46, knn, measured_samples)
    currdata = Construct_Currdata(measured_samples, 47, 46)
    
    total = len(currdata) * len(currdata[0])

    knownnum = CountKnownNum(currdata)
    for i in range(len(currdata)):
        for j in range(len(currdata[0])):
            #if  i == j or (i == 92 and j == 91) or (i == 93 and j == 91):
            if i == 0 or j == 0:
                '''
                row = (i/2 if i % 2 == 0 else (i - 1)/2)
                col = (j/2 if j % 2 == 0 else (j - 1)/2)
                '''
                row = (i - 47 if i > 46 else i)
                col = (j - 46 if j > 45 else j)
                currdata[i][j] = ground_truth10[i][j]
                if (row, col) in measured_samples_:
                    measured_samples_[(row, col)] += 1
                else: measured_samples_[(row, col)] = 1
    pd_data = pd.DataFrame(currdata)
    pd_data.to_csv('initial_data.csv',index=False,header=False)
    clusters_num = []
    P = 1
    batch_size = int(total*0.01)
    accus = {}
    accusReggression = []

    vectors = []
    new_added_num = []

    current_labels = measured_samples[:,-1].copy().astype(np.int16)
    clusters_num.append(len(np.unique(current_labels))/len(np.unique(clustering.labels_)))
#-------------------------------------------------------------------------------------------------------------------------
#score
    #currdata = Cons(ground_truth10, 0.02)
    knownnum = CountKnownNum(currdata)
    total = len(currdata) * len(currdata[0])
    zeros = total - knownnum
    once = 0
    twice = 0
    thriple = 0
    quartic = 0
    for key in measured_samples_:
        if measured_samples_[key] == 1:
            once += 1
        elif measured_samples_[key] == 2:
            twice += 1
        elif measured_samples_[key] == 3:
            thriple += 1
        elif measured_samples_[key] == 4:
            quartic += 1
    zeros = total/4 - once - twice - thriple - quartic
    vectors.append((zeros*4/total, once*4/total, twice*4/total, thriple*4/total, quartic*4/total))
    
    while knownnum < total:
        #Print_average_col_score(currdata)
        step = 1
        panelty = P * (knownnum/total) + 1
        prediction_temp = Simulate_Once(currdata, step, panelty)
        curr_resp, resp_average = Calculate_Resp(currdata, panelty)

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
            for label in dictp:
                if dictp[label] > maxscore:
                    maxlabel = label
                    maxscore = dictp[label]
            '''
            if maxscore < curr_resp[row][1]:
                maxlabel = curr_resp[row][0]
            '''
            if ground_truth10[row][col] == maxlabel:
                    correct+=1
                    corrects.append(maxscore)
            else:
                mistake.append(maxscore)
            #TF_ratio = Calculate_TrueFalsth_Ratio(prediction_temp[key], maxlabel)
            bias = 1 - min(dictp.values())
            for key in dictp:
                dictp[key] += bias
            k = 1 / sum(dictp.values())
            for key in dictp:
                dictp[key] *= k
            uncertain_score = Calculate_entropy(dictp)
            predictions.append([row, col, maxlabel, maxscore, uncertain_score])
        #print('correct mean: ', np.array(corrects).mean())
        #print('mistake mean: ', np.array(mistake).mean())
        
        accu = 1 - (count - correct)/total
        #accu = correct/count
        accus[knownnum] = accu
        

        size = min(batch_size, total - knownnum)
        #predictions = np.array(predictions)
        batch = []
        Make_Selections_random(int(size), predictions, batch)
        #Make_Selections_Higher_Score(max(1, int(size/2)), predictions, batch)
        new_added = []
        for i in range(len(batch)):
            currdata[batch[i][0]][batch[i][1]] = ground_truth10[batch[i][0]][batch[i][1]]
            
            if batch[i][0] > 46:
                row = batch[i][0] - 47
            else: row = batch[i][0]
            if batch[i][1] > 45:
                col = batch[i][1] - 46
            else: col = batch[i][1]
            '''
            row = (batch[i][0]/2 if batch[i][0] % 2 == 0 else (batch[i][0] - 1)/2 )
            col = (batch[i][1]/2 if batch[i][1] % 2 == 0 else (batch[i][1] - 1)/2 )
            '''
            batch[i] = (row, col)
            if (row, col) in measured_samples_:
                measured_samples_[(row, col)] += 1
            else: 
                measured_samples_[(row, col)] = 1
                new_added.append((row, col))
        new_added_num.append(len(new_added))
        for sample in XY:
            row = sample[1]-1
            col = sample[0]-1
            if (row, col) in new_added:
                measured_samples = np.vstack((measured_samples, sample))
        knownnum = CountKnownNum(currdata)
        once = 0
        twice = 0
        thriple = 0
        quartic = 0
        for key in measured_samples_:
            if measured_samples_[key] == 1:
                once += 1
            elif measured_samples_[key] == 2:
                twice += 1
            elif measured_samples_[key] == 3:
                thriple += 1
            elif measured_samples_[key] == 4:
                quartic += 1
        zeros = total/4 - once - twice - thriple - quartic
        vector = (zeros*4/total, once*4/total, twice*4/total, thriple*4/total, quartic*4/total)
        vectors.append(vector)
        accuRg = 0.42 * vector[0] + 1 * vector[1] - 0.57 * vector[2] + 6.4 * vector[3] - 21 * vector[4]
        accusReggression.append(accuRg)
        #accus[knownnum] = accu
        #measured_samples = np.array(measured_samples_)
        
        current_labels = measured_samples[:,-1].copy().astype(np.int16)
        knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance').fit(measured_samples[:, 4:-1], current_labels)
        clusters_num.append(len(np.unique(current_labels))/len(np.unique(clustering.labels_)))

        #currdata = Construct_Currdata(measured_samples, measured_y, 47, 46)
        ground_truth10 = Update_Groundtruth(XY, 47, 46, knn, measured_samples)
        #pd_data = pd.DataFrame(ground_truth10)
        #pd_data.to_csv('groundtruth.csv',index=False,header=False)
        currdata = Update_currdata(currdata, ground_truth10)


        '''
        for selection in selections:
            row = selection[0]
            col = selection[1]
            currdata[row][col] = ground_truth10[row][col]

        
        for selection in selections2:
            row = selection[0]
            col = selection[1]
            currdata[row][col] = ground_truth10[row][col]
        knownnum = CountKnownNum(currdata)
        '''
        '''
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
    data = np.array(Readin_xls('E:\Active_Learning\elife\elife-10047-supp4-v2.xls'))

    measured_samples_ = []
    measured_initial = []
   
    for sample in data:
        row = sample[1]-1
        col = sample[0]-1
        if row == col or (46 - row) == col:
            measured_samples_.append((row, col))
            measured_initial.append(sample)
    measured_samples = np.array(measured_initial)

    #clustering = AgglomerativeClustering(distance_threshold=5, n_clusters=None).fit(measured_samples[:, 4:])
    clustering = AgglomerativeClustering(n_clusters=60).fit(measured_samples[:, 4:])
    measured_y = clustering.labels_


    knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance').fit(measured_samples[:, 4:], measured_y)
    ground_truth10 = Construct_Groundtruth(data, 47, 46, knn)
    currdata = Construct_Currdata(measured_samples, 47, 46)
    
    total = len(currdata) * len(currdata[0])

    knownnum = CountKnownNum(currdata)
    for i in range(len(currdata)):
        for j in range(len(currdata[0])):
            if i == j or (93 - i) == j:
                currdata[i][j] = ground_truth10[i][j]
    pd_data = pd.DataFrame(currdata)
    pd_data.to_csv('initial_data.csv',index=False,header=False)
    batch_size = int(0.01 * total)
    

    while knownnum < total:
        #Print_average_col_score(currdata)
        

        step = 1
        panelty = P * (knownnum/total) + 1
        prediction_temp = Simulate_Once(currdata, step, panelty)

        curr_resp, resp_average = Calculate_Resp(currdata, panelty)

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
            for label in dictp:
                if dictp[label] > maxscore:
                    maxlabel = label
                    maxscore = dictp[label]

            
            if maxscore < curr_resp[row][1]:
                maxlabel = curr_resp[row][0]
            
            if ground_truth10[row][col] == maxlabel:
                    correct+=1
                    corrects.append(maxscore)
            else:
                mistake.append(maxscore)
            #TF_ratio = Calculate_TrueFalsth_Ratio(prediction_temp[key], maxlabel)
            bias = 1 - min(dictp.values())
            for key in dictp:
                dictp[key] += bias
            k = 1 / sum(dictp.values())
            for key in dictp:
                dictp[key] *= k
            uncertain_score = Calculate_entropy(dictp)
            predictions.append([row, col, maxlabel, maxscore, uncertain_score])
        #print('correct mean: ', np.array(corrects).mean())
        #print('mistake mean: ', np.array(mistake).mean())
        #print()
        knownnum = CountKnownNum(currdata)

        accu = 1 - (count - correct)/total
        #accu = correct/count
        accusR[knownnum] = accu

        size = min(batch_size, total - knownnum)
        predictions = np.array(predictions)
        if Generate_binary_result(0.5):
            selections = Make_Selections_Lower_Score(size, predictions)
            for p in predictions:
                predictions = predictions.delete(p)
            selections2 = Make_Selections_Higher_Score(0, predictions)
        else:
            selections = Make_Selections_Lower_Score(0, predictions)
            for p in selections:
                predictions = predictions.delete(p)
            selections2 = Make_Selections_Higher_Score(size, predictions)

        new_added = []
        for i in range(len(selections)):
            currdata[int(selections[i][0])][int(selections[i][1])] = ground_truth10[int(selections[i][0])][int(selections[i][1])]
            if selections[i][0] > 46:
                row = int(selections[i][0]) - 47
            else : row = int(selections[i][0])
            if selections[i][1] > 45:
                col = int(selections[i][1]) - 46
            else: col = int(selections[i][1])
            if (row, col) not in measured_samples_:
                new_added.append((row, col))
                measured_samples_.append((row, col))
        for i in range(len(selections2)):
            currdata[int(selections[i][0])][int(selections[i][1])] = ground_truth10[int(selections[i][0])][int(selections[i][1])]
            if selections2[i][0] > 46:
                row = int(selections2[i][0]) - 47
            else : row = int(selections2[i][0])
            if selections2[i][1] > 45:
                col = int(selections2[i][1]) - 46
            else: col = int(selections2[i][1])
            if (row, col) not in measured_samples_:
                new_added.append((row, col))
                measured_samples_.append((row, col))

        for sample in data:
            row = sample[1]-1
            col = sample[0]-1
            if (row, col) in new_added:
                measured_samples = np.vstack((measured_samples, sample))
        #measured_samples = np.array(measured_samples_)
        #clustering = AgglomerativeClustering(distance_threshold=5, n_clusters=None).fit(measured_samples[:, 4:])
        clustering = AgglomerativeClustering(n_clusters=60).fit(measured_samples[:, 4:])

        clusters_num.append(clustering.n_clusters_)
        measured_y = clustering.labels_


        #forest = RandomForestClassifier(n_estimators=int(0.1 * len(measured_samples)), max_depth = 7,
#                                        max_features=100).fit(measured_samples[:, 4:], measured_y)
        knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance').fit(measured_samples[:, 4:], measured_y)

        #currdata = Construct_Currdata(measured_samples, measured_y, 47, 46)
        ground_truth10 = Construct_Groundtruth(data, 47, 46, knn)
        #pd_data = pd.DataFrame(ground_truth10)
        #pd_data.to_csv('groundtruth.csv',index=False,header=False)
        currdata = Update_currdata(currdata, ground_truth10)
        '''


    print('uniquesness: ', uniqueness, 'responsiveness: ', responsiveness)
    #print(accus.values())
    print(' ')
    print(' ')
    '''
    range1 = np.array(accus.keys())
    range1 /= total
    range2 = np.array(accusR.keys())
    range2 /= total
    '''
    #plt.plot(accusHybrid.keys(), accusHybrid.values(), 'o-', label='active learning(uncertainty score)')
    plt.plot(accus.keys(), accus.values(), 'o-', label='active learning(uncertainty score)')
    #plt.plot(accusEntropy.keys(), accusEntropy.values(), 'o-', label='active learning(entropy)')
    #plt.plot(accusR.keys(), accusR.values(), 'o-', label = 'random learning')
    
    
    plt.legend(loc='best')
    plt.show()

    plt.plot(np.arange(len(clusters_num))+1, clusters_num, '^-', label = 'percentage of discovered phenotypes')
    plt.legend(loc='best')
    plt.xlabel('round')
    plt.show()
    vectors = np.array(vectors)
    plt.plot(np.arange(len(vectors)), vectors[:, 0], label='unmeasured')
    plt.plot(np.arange(len(vectors)), vectors[:, 1], label='measured once')
    plt.plot(np.arange(len(vectors)), vectors[:, 2], label='measured twice')
    plt.plot(np.arange(len(vectors)), vectors[:, 3], label='measured 3 times')
    plt.plot(np.arange(len(vectors)), vectors[:, 4], label='measured 4 times')
    plt.legend(loc='best')
    plt.title('quads discover rate')
    plt.show()


    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.plot(np.arange(len(accusReggression)), accusReggression, label = 'Reggression accuracy, not real accuracy')
    plt.show()
    

    filename = 'test-new1.csv'
    result = []
    result.append(accus.keys())
    #result.append(accusHybrid.values())
    result.append(accus.values())
    #result.append(accusEntropy.values())
    #result.append(accusR.values())
    
 
    
    pd_data = pd.DataFrame(result)
    pd_data.to_csv(filename,index=False,header=False)

    
def main():
    Trail_ActiveLearning(0, 0, 0.02)


if __name__ == "__main__":
    main()

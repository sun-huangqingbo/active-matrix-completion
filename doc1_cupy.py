import os
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import math
import multiprocessing



def Preparing_Labels(labelnums):
    labels = []
    for i in range(0, labelnums):
        strtemp = str(i)
        labels.append(strtemp)
    return labels

def Generate_binary_result(prob):
    rand = np.random.uniform()
    if rand < prob:
        return True
    else: return False



def Generating_Truth(uniqueness, responsiveness, numtypes):
    labels = Preparing_Labels(numtypes)
    
    table = []


    
    for i in range(40):
        row = []
        rand = np.random.randint(len(labels))
        row.append(labels[rand])
        table.append(row)

    for i in range(40):
        rest_pool = labels.copy()
        rest_pool.remove(table[i][0])
        for j in range(39):
            if Generate_binary_result(responsiveness):
                table[i].append(rest_pool[np.random.randint(len(rest_pool))])
            else:
                table[i].append(table[i][0])

    for j in range(60):
        rand = np.random.randint(40)
        for i in range(40):
            table[i].append(table[i][rand])

    for i in range(60):
        table.append(table[np.random.randint(40)])


    table = np.array(table)
    #table = cp.array(table)
    

    pd_data = pd.DataFrame(table)
    pd_data.to_csv('test_data.csv',index=False,header=False)
    
    return table





'''


def Generating_Truth(targetsnum, conditionsnums, percent):
    labelsnum = int(targetsnum * conditionsnums * percent + 1)
    labels = Preparing_Labels(labelsnum)
    ground_truth = []
    for i in range(0, targetsnum * conditionsnums):
        ground_truth.append(labels[np.random.randint(labelsnum)])
    
    ground_truth = np.array(ground_truth).reshape(targetsnum, conditionsnums)

    return ground_truth
'''
def Initialize_Data(ground_truth, observed_percent):
    observednum = int(len(ground_truth[0]) * len(ground_truth) * observed_percent)
    currdata = []

    for i in range(0, len(ground_truth) * len(ground_truth[0])):
        currdata.append('u')

    currdata = np.array(currdata).reshape(len(ground_truth), len(ground_truth[0]))

    for i in range(observednum):
        rand1 = np.random.randint(len(ground_truth))
        rand2 = np.random.randint(len(ground_truth[0]))
        while currdata[rand1][rand2] != 'u':
            rand1 = np.random.randint(len(ground_truth))
            rand2 = np.random.randint(len(ground_truth[0]))
        currdata[rand1][rand2] = ground_truth[rand1][rand2]
        
    return currdata


def Calculate_Similarity(currdata, col1, col2):
    same = 0
    for i in range(len(currdata)):
        if currdata[i][col1] != 'u':
            if currdata[i][col2] != 'u':
                if currdata[i][col1] == currdata[i][col2]:
                    same = same + 1
    
    return same

'''
def Predict_One_Entry(row, col, currdata):
    potential_conditions = []
    for col in range(len(currdata[0])):
        if currdata[row][col] != 'u':
            potential_conditions.append(col)
    
    for col in potential_conditions:
        
'''




def HasCoobservation(col, col2, rows, currdata):
    for row in rows:
        if currdata[row][col] == currdata[row][col2] and currdata[row][col] != 'u':
            return True
    
    return False



def CalculateClusterScore(cols, rows, currdata):
    score = 0
    for row in rows:
        for col in cols:
            if currdata[row][col] != "u":
                score+=1

    return score



def Predict_One_Entry(row, col, current_data):
    currdata = current_data.copy()
    #print("current data:\n", currdata)
    potential_labels = []
    for e in currdata[row]:

        if e != "u":
            if e not in potential_labels:
                potential_labels.append(e)


    prediction = 'u'
    num = -1



    label_prediction = {}
    for label in potential_labels:
        #print(label, ":")
        currdata[row][col] = label #assuming 
        elementsNum = -1
        
        potential_cols = []

        for j in range(len(currdata[0])):
            if (currdata[row][j] == label or currdata[row][j] == 'u') and j != col:
                potential_cols.append(j)
#--------------------------------------------------------------------------------------------------
# if each col in potential_cols has at least one coobservation with others
        '''
        for col_ in potential_cols:
            othercols = potential_cols.copy()
            othercols.remove(col_)
            othercols.append(col)

            coobervation = False
            for col2 in othercols:
                if HasCoobservation(col_, col2, currdata):
                    coobervation = True
                    break

            if not coobervation:
                potential_cols.remove(col_)
                '''

#--------------------------------------------------------------------------------------------------
# initialize the cluster
        cluster_cols = []
        cluster_cols.append(col)
        cluster_score = 0
        cluster_rows = []
        for i in range(len(currdata)):
            cluster_rows.append(i)
            if currdata[i][col] != "u":
                cluster_score += 1
#--------------------------------------------------------------------------------------------------
# start to adding cols into cluster
        no_added = False
        while not no_added:

            cols_score = []
            for k in range(len(potential_cols)):
                updated_cluster_score = 0
                potetial_score = 0




                rest_potential_cols = potential_cols.copy()
                rest_potential_cols.remove(potential_cols[k])
## assume add col to cluster
                updated_cluster_cols = cluster_cols.copy()
                updated_cluster_cols.append(potential_cols[k])
                updated_cluster_rows = cluster_rows.copy()
                
                for row_ in updated_cluster_rows:
                    row_labels = []
                    tag = True
                    for col2 in updated_cluster_cols:
                        if currdata[row_][col2] != "u":
                            row_labels.append(currdata[row_][col2])
                    if len(row_labels) > 0:
                        e0 = row_labels[0]
                        for e in row_labels:
                            if e != e0:
                                tag = False
                                break
                        if not tag:
                            updated_cluster_rows.remove(row_)
#--------------------------------------------------------------------------------------------------
# judge if col_ has at least one coobservation with col2 in cluster
                
                coobervation = False
                for col2 in cluster_cols:
                    if HasCoobservation(potential_cols[k], col2, updated_cluster_rows, currdata):
                        coobervation = True
                        break

                if not coobervation:
                    continue
 
    
## calculate the score
                updated_cluster_score = CalculateClusterScore(updated_cluster_cols, updated_cluster_rows, currdata)
                potential_score = CalculateClusterScore(rest_potential_cols, cluster_rows, currdata)
                col_score = (k, updated_cluster_score, potential_score)
                cols_score.append(col_score)

            col_index = []
            max_score = -1
            for col_score in cols_score:
                if col_score[1] > cluster_score:
                    if (col_score[1] + col_score[2]) > max_score:
                        max_score = (col_score[1] + col_score[2])
                        col_index.clear()
                        col_index.append(col_score[0])
                    elif (col_score[1] + col_score[2]) == max_score:
                        col_index.append(col_score[0])

            if len(col_index) > 1:
                # add col
                col_for_adding = np.random.randint(len(col_index))
                cluster_cols.append(potential_cols[col_index[col_for_adding]])
                potential_cols.remove(potential_cols[col_index[col_for_adding]])
                for row_ in cluster_rows:
                    row_labels = []
                    tag = True
                    for col2 in cluster_cols:
                        if currdata[row_][col2] != "u":
                            row_labels.append(currdata[row_][col2])
                 
                    if len(row_labels) > 0:
                        e0 = row_labels[0]
                        for e in row_labels:
                            if e != e0:
                                tag = False
                                break
                        if not tag:
                            cluster_rows.remove(row_)
                cluster_score = CalculateClusterScore(cluster_cols, cluster_rows, currdata)
            else: no_added =True

#------------------------------------------------------------------------------------------------
#print cluster


#            for row_ in cluster_rows:
#                print()
#                for col_ in cluster_cols:
#                    print(currdata[row_][col_], ", ", end='')

#            print("done!")
           


#------------------------------------------------------------------------------------------------
#save prediction

        
        label_prediction[label] = cluster_score



    '''
    predicted_label = ""
    support_score = -1
    for prediction in label_prediction:
        if prediction[1] > support_score:
            predicted_label = prediction[0]
            support_score = prediction[1]
    '''

    return (label_prediction)



def SimulationOnce(ground_truth, currdata):
    

    predictions = []


    for row in range(len(currdata)):
        for col in range(len(currdata[row])):
            if currdata[row][col] == 'u':
                prediction = Predict_One_Entry(row, col, currdata)
                currdataT = currdata.T
                prediction_row_cluster = Predict_One_Entry(col, row, currdataT)
                for key in prediction_row_cluster:  #key is the label, value is the score
                    if key in prediction:
                        prediction[key] += prediction_row_cluster[key]
                    else:
                        prediction[key] = prediction_row_cluster[key]
                predictions.append((row, col, prediction))


def Simulation_Parallel(currdata):
    unknown_entries = []
    for row in range(len(currdata)):
        for col in range(len(currdata[row])):
            if currdata[row][col] == 'u':
                unknown_entries.append((row, col, currdata))

    core_num = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(core_num)
    predictions = pool.map(Split_parameters, unknown_entries)
    pool.close()
    pool.join()
    return predictions


def Split_parameters(paramslist):
    row = paramslist[0]
    col = paramslist[1]
    currdata = paramslist[2]
    return Predict_two_times(row, col, currdata)

def Predict_two_times(row, col, currdata):
    prediction = Predict_One_Entry(row, col, currdata)
    currdataT = currdata.T
    prediction_row_cluster = Predict_One_Entry(col, row, currdataT)
    for key in prediction_row_cluster:  #key is the label, value is the score
        if key in prediction:
            prediction[key] += prediction_row_cluster[key]
        else:
            prediction[key] = prediction_row_cluster[key]
    return (row, col, prediction)


def CountKnownNum(currdata):
    count = 0
    for row in currdata:
        for e in row:
            if e != 'u':
                count+=1
    return count

def Make_Selections_entropy(num, predictions):
    batch = []
    for prediction in predictions:
        if len(batch) < num:
            batch.append(prediction)  #prediction format (row, col, label, score)
        else:
            min_entropy = batch[0][3]
            index = 0
            for i in range(len(batch)):
                if batch[i][3] < min_entropy:
                    min_entropy = batch[i][3]
                    index = i

            if prediction[3] > min_entropy:
                batch.remove(batch[index])
                batch.append(prediction)


    return batch

def Make_Selections_score(num, predictions):
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


def Make_Selections_random(num, predictions):
    predictions_copy = predictions.copy()
    batch = []
    for k in range(num):
        rand = np.random.randint(len(predictions_copy))
        batch.append(predictions_copy[rand])
        del(predictions_copy[rand])


    return batch

def Calculate_Entropy(prediction):
    prob = []
    total_score = 0
    entropy = 0
    for key in prediction[2]:
        total_score += prediction[2][key]
    for key in prediction[2]:
        prob =  prediction[2][key]/total_score
        entropy -= prob * math.log2(prob)

    return entropy


def Trail(batch_size, initailize_percent):
    ground_truth = Generating_Truth(0, .8, 8)
    total = len(ground_truth) * len(ground_truth[0])

    currdata = Initialize_Data(ground_truth, initailize_percent)
    scores = {}
    known_percent = []
    knownnum = CountKnownNum(currdata)
    while knownnum < total:
        knownnum = CountKnownNum(currdata)
        #batch_size = 40
        size = min(batch_size, total - knownnum)

        entropy_correct = []
        score_correct = []
        entropy_false = []
        score_false = []

        predictions = Simulation_Parallel(currdata)
        count = len(predictions)
        correct = 0
        local_prediction = []
        for prediction in predictions:
            maxscore = -1
            maxlabel = ''
            for label in prediction[2]:
                if prediction[2][label] > maxscore:
                    maxlabel = label
                    maxscore = prediction[2][label]

            entropy = Calculate_Entropy(prediction)

            local_prediction.append((prediction[0], prediction[1], maxlabel, maxscore))  #score or entropy
            

            if ground_truth[prediction[0]][prediction[1]] == maxlabel:
                correct+=1
                score_correct.append(maxscore)
                entropy_correct.append(entropy)
            else:
                score_false.append(maxscore)
                entropy_false.append(entropy)


        score_correct_mean = cp.array(score_correct).mean()
        score_false_mean = cp.array(score_false).mean()
        entropy_correct_mean = cp.array(entropy_correct).mean()
        entropy_false_mean = cp.array(entropy_false).mean()
        '''
        plt.figure(figsize=(12, 20))
        ax1 = plt.subplot(211)
        ax1.plot(range(len(score_correct)), score_correct)
        ax1.plot(range(len(score_false)), score_false)
        ax2 = plt.subplot(212)
        ax2.plot(range(len(entropy_correct)), entropy_correct)
        ax2.plot(range(len(entropy_false)), entropy_false)
        plt.show()
        '''

        scores[knownnum] = correct/count
        selections = Make_Selections_score(size, local_prediction)

        for selection in selections:
            row = selection[0]
            col = selection[1]

            currdata[row][col] = ground_truth[row][col]

        knownnum = CountKnownNum(currdata)
    
    return scores



def Trail_entropy(batch_size, initailize_percent):
    ground_truth = Generating_Truth(10, 6, 0)
    total = len(ground_truth) * len(ground_truth[0])

    currdata = Initialize_Data(ground_truth, initailize_percent)
    scores = {}
    known_percent = []
    knownnum = CountKnownNum(currdata)
    while knownnum < total:
        knownnum = CountKnownNum(currdata)
        #batch_size = 40
        size = min(batch_size, total - knownnum)


        
        #known_percent.append(knownnum/total)


        
        predictions = SimulationOnce(ground_truth, currdata)
        count = len(predictions)
        correct = 0
        local_prediction = []
        for prediction in predictions:
            maxscore = -1
            maxlabel = ''
            for label in prediction[2]:
                if prediction[2][label] > maxscore:
                    maxlabel = label
                    maxscore = prediction[2][label]

            entropy = Calculate_Entropy(prediction)

            local_prediction.append((prediction[0], prediction[1], maxlabel, entropy))  #score or entropy
            

            if ground_truth[prediction[0]][prediction[1]] == maxlabel:
                correct+=1

        scores[knownnum] = correct/count
        selections = Make_Selections_entropy(size, local_prediction)

        for selection in selections:
            row = selection[0]
            col = selection[1]

            currdata[row][col] = ground_truth[row][col]

        knownnum = CountKnownNum(currdata)
    
    return scores




def Trail_random(batch_size, initailize_percent):
    ground_truth = Generating_Truth(0, .8, 8)
    total = len(ground_truth) * len(ground_truth[0])

    currdata = Initialize_Data(ground_truth, initailize_percent)
    scores = {}
    known_percent = []
    knownnum = CountKnownNum(currdata)
    while knownnum < total:
        knownnum = CountKnownNum(currdata)
        #batch_size = 40
        size = min(batch_size, total - knownnum)


        
        #known_percent.append(knownnum/total)


        
        predictions = SimulationOnce(ground_truth, currdata)
        count = len(predictions)
        correct = 0
        local_prediction = []
        for prediction in predictions:
            maxscore = -1
            maxlabel = ''
            for label in prediction[2]:
                if prediction[2][label] > maxscore:
                    maxlabel = label
                    maxscore = prediction[2][label]

            #entropy = Calculate_Entropy(prediction)

            local_prediction.append((prediction[0], prediction[1], maxlabel, maxscore))  #score or entropy
            

            if ground_truth[prediction[0]][prediction[1]] == maxlabel:
                correct+=1

        scores[knownnum] = correct/count
        selections = Make_Selections_random(size, local_prediction)

        for selection in selections:
            row = selection[0]
            col = selection[1]

            currdata[row][col] = ground_truth[row][col]

        knownnum = CountKnownNum(currdata)
    
    return scores


def main():
    if __name__ == "__main__":
        np.random.seed()
        accuracy_score = {}
        accuracy_entropy = {}
        accuracy_random = {}
        num_of_trails = 1

        batch_size = 1000
        initialize_percent = .6

        for k in range(num_of_trails):
            score_in_one_trail = Trail(batch_size, initialize_percent)
            for percent in score_in_one_trail:
                if percent in accuracy_score:
                    accuracy_score[percent] = accuracy_score[percent]+score_in_one_trail[percent]
                else:
                    accuracy_score[percent] = score_in_one_trail[percent]

            '''

            score_in_one_trail = Trail_entropy(batch_size, initialize_percent)
            for percent in score_in_one_trail:
                if percent in accuracy_entropy:
                    accuracy_entropy[percent] = accuracy_entropy[percent]+score_in_one_trail[percent]
                else:
                    accuracy_entropy[percent] = score_in_one_trail[percent]
            
            score_in_one_trail = Trail_random(batch_size, initialize_percent)
            for percent in score_in_one_trail:
                if percent in accuracy_random:
                    accuracy_random[percent] = accuracy_random[percent]+score_in_one_trail[percent]
                else:
                    accuracy_random[percent] = score_in_one_trail[percent]
        '''
        for key in accuracy_score:
            accuracy_score[key] = accuracy_score[key]/num_of_trails

        #for key in accuracy_entropy:
            #accuracy_entropy[key] = accuracy_entropy[key]/num_of_trails
        
        #for key in accuracy_random:
            #accuracy_random[key] = accuracy_random[key]/num_of_trails


        percent_score = []
        for key in accuracy_score.keys():
            percent_score.append(key/10000)
        '''
        percent_entropy = []
        for key in accuracy_entropy.keys():
            percent_entropy.append(key/480)
        
        percent_random = []
        for key in accuracy_random.keys():
            percent_random.append(key/480)
        '''

        plt.plot(percent_score, accuracy_score.values(), "-", label="uncertainy: # of observed samples")
        #plt.plot(percent_entropy, accuracy_entropy.values(), "-", label="uncertainy: entropy")
        #plt.plot(percent_random, accuracy_random.values(), "-", label="random chosing")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.show()






main()
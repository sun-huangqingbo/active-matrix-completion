import multiprocessing
import numpy as np
import scipy
import copy



class Categorical_Matrix_Solver:
    def __init__(self, batch_size, missing_value_term, query_strategy='hybrid', core_num=1, confidence_correction=True):
        self.batch_size = batch_size
        self.strategy = query_strategy
        self.core_num = core_num
        self.missing_value_term = missing_value_term
        self._fit = False
        self.adjustment = confidence_correction
        self._batch = []
    
    def query(self):
        if not self._fit:
            raise AssertionError('Must fit the solver at first.') 
        else: 
            return self._batch


    def fit_transform(self, matrix):
        currdata = copy.deepcopy(matrix)
        if type(currdata).__module__ != np.__name__:
            currdata = np.array(currdata)
        if len(currdata.shape) != 2:
            raise ValueError('The dimension of input matrix is not 2.')
        miss_num = 0
        for i in range(currdata.shape[0]):
            for j in range(currdata.shape[1]):
                if currdata[i][j] == self.missing_value_term:
                    currdata[i][j] = -1
                    miss_num += 1
        total = len(currdata) * len(currdata[0])
        knownnum = self._count_known_num(currdata)
        step = 1
        panelty =  2.5 * (knownnum/total) 
        prediction_temp = self._simulate_once(currdata, step, panelty)
        curr_resp, resp_average = self._calculate_resp(currdata, panelty)
        row_common_label_ = self._row_common_label(currdata)
        global_common_label_ = self._global_common_label(currdata)
        predictions = []
        for key in prediction_temp:
            row = key[0]
            col = key[1]
            dictp = prediction_temp[key]
            maxscore = -1000000
            maxlabel = np.nan
            if dictp != -2 :
                for label in dictp:
                    if dictp[label] > maxscore:
                        maxlabel = label
                        maxscore = dictp[label]
            else:
                if row_common_label_[row] != -2:
                    maxlabel = curr_resp[row][0]
                    maxscore = -panelty * (knownnum/total)
                else:
                    maxlabel = global_common_label_
                    maxscore = -panelty * (knownnum/total)
            if self.adjustment:
                if maxscore < curr_resp[row][1] and curr_resp[row][0] != -2:
                    maxlabel = curr_resp[row][0]
            if dictp != -2:
                bias = 1 - min(dictp.values())
                for key in dictp:
                    dictp[key] += bias
                k = 1 / sum(dictp.values())
                for key in dictp:
                    dictp[key] *= k
                uncertain_score = self._calculate_entropy(dictp)
            else: uncertain_score = 1
            predictions.append([row, col, maxlabel, maxscore, uncertain_score])
        for p in predictions:
            row = int(p[0])
            col = int(p[1])
            currdata[row][col] = p[2]
        size = min(miss_num, self.batch_size)
        if self.strategy == 'hybrid':
            selections = self._make_selections_low_score(size*0.5, predictions)
            for p in selections:
                predictions.remove(p)
            for p in predictions:
                p[3] = p[4]
            selections2 = self._make_selections_high_score(size - int(self.batch_size*0.5), predictions)
        elif self.strategy == 'entropy':
            selections = []
            for p in selections:
                predictions.remove(p)
            for p in predictions:
                p[3] = p[4]
            selections2 = self._make_selections_high_score(size, predictions)
        elif self.strategy == 'score':
            selections = self._make_selections_low_score(size, predictions)
            selections2 = []
        elif self.strategy == 'random':
            selections = self._make_selections_random(size, predictions)
            selections2 = []
        else: raise ValueError('unknown query strategy.')
        self._batch.clear()
        for selection in selections:
            row = selection[0]
            col = selection[1]
            self._batch.append((row, col))
        for selection in selections2:
            row = selection[0]
            col = selection[1]
            self._batch.append((row, col))
        self._fit = True
        return currdata

    def _count_known_num(self, currdata):
        count = 0
        for row in currdata:
            for e in row:
                if e != -1:
                    count+=1
        return count

    def _make_selections_high_score(self, num, predictions):
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

    def _make_selections_low_score(self, num, predictions):
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

    def _predict_one_col(self, col, currdata, step, panelty):
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
            community = self._find_cols_with_n_conflicts(n_conflicts + step, n_conflicts, currdata, col)
            if len(community) != 0:
                for urow in unknown_rows:
                    prediction = {}
                    count = {}
                    for col2 in community:
                        if currdata[urow][col2] != -1:
                            label = currdata[urow][col2]
                            if label in prediction:
                                prediction[label] += self._count_col_score(col, col2, panelty, currdata)
                                count[label] += 1
                            else:
                                prediction[label] = self._count_col_score(col, col2, panelty, currdata)
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

    def _count_col_score(self, col, col2, panelty, currdata):
        score = 0
        for row in currdata:
            if row[col] != -1 and row[col2] != -1 and row[col] == row[col2]:
                score += 1
            if row[col] != -1 and row[col2] != -1 and row[col] != row[col2]:
                score -= panelty
        return score

    def _find_cols_with_n_conflicts(self, upperbound, lowerbound, currdata, col):
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

    def _split_parameters(self, pl):
        col = pl[0]
        currdata = pl[1]
        step = pl[2]
        panelty = pl[3]
        return self._predict_one_col(col, currdata, step, panelty)

    def _simulate_once(self, currdata, step, panelty):
        prediction_from_col = []
        prediction_from_row = []
        predictions = {}
        parameters = []
        for col in range(len(currdata[0])):
            parameters.append((col, currdata, step, panelty))

        core_num = min(multiprocessing.cpu_count(), self.core_num)
        pool = multiprocessing.Pool(core_num)
        predictions_col = pool.map(self._split_parameters, parameters)
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
        predictions_row = pool.map(self._split_parameters, parameters)
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


    def _make_selections_random(self, num, predictions):
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

    def _calculate_resp(self, currdata, panelty):
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

    def _calculate_entropy(self, prediction):
        entropy = 0
        for key in prediction:
            p = prediction[key]
            entropy -= p * np.lib.scimath.log2(p)
        return entropy

    def _row_common_label(self, mtx):
        label_set = []
        for i in range(len(mtx)):
            temp = {}
            for j in range(len(mtx)):
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

    def _global_common_label(self, mtx):
        temp = {}
        for i in range(len(mtx)):
            for j in range(len(mtx)):
                label = mtx[i][j]
                if label != -1 and label not in temp:
                    temp[label] = 1
                elif label != -1 and label in temp:
                    temp[label] += 1
            temp_sorted = sorted(temp.items(), key=lambda x:x[1], reverse = True)
        return temp_sorted[0][0]


import pandas as pd
import matplotlib.pyplot as plt
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
    # filename = str(uniqueness) + 'test_data.csv'
    # pd_data = pd.DataFrame(table)
    # pd_data.to_csv('test_data.csv',index=False,header=False)
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
    return np.array(currdata)


ground_truth10 = Generating_Truth(100, 0.4, 0.8, 32)
currdata = Initialize_Data(ground_truth10, 100)
total = len(currdata) * len(currdata[0])
knownnum = CountKnownNum(currdata)
P = 2.5
batch_size = 100
accus = {}
if __name__ == "__main__":
    solver = Categorical_Matrix_Solver(100, -1, core_num=10)
    while knownnum < total:
        recover = solver.fit_transform(currdata)
        batch = solver.query()
        mistake = 0
        for i in range(recover.shape[0]):
            for j in range(recover.shape[1]):
                if currdata[i][j] == -1 and recover[i][j] != ground_truth10[i][j]:
                    mistake += 1
        accus[knownnum] = 1 - mistake/total
        for item in batch:
            row = item[0]
            col = item[1]
            currdata[row][col] = ground_truth10[row][col]
        knownnum = CountKnownNum(currdata)
    plt.plot(list(accus.keys()), list(accus.values()), 'o-', label=  'active learning')
    plt.show()



    

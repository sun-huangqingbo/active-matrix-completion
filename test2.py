import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import multiprocessing
import csv
import scipy
import argparse
import random, copy 

def Count_Col_Score(col, col2, panelty, currdata):
    score = 0
    same = 0
    conflict = 1
    for row in currdata:
        if row[col] != -1 and row[col2] != -1 and row[col] == row[col2]:
            # score += 1
            same += 1
        if row[col] != -1 and row[col2] != -1 and row[col] != row[col2]:
            # score -= panelty
            conflict+=1
    return same/conflict

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


np.random.seed(0)
ground_truth10 = np.load("experimental-space-v10.npy") 
row_dict = {}
row_delete = []
for i in range(len(ground_truth10)):
    for j in range(200):
        committee = Find_Cols_With_n_Conflicts(j+1, j, ground_truth10.T, i)
        if len(committee) > 0:
            row_dict[i] = j
            if j > 20:
                row_delete.append(i)
            break
ground_truth10 = np.delete(ground_truth10, row_delete, axis=0)
col_dict = {}
col_delete = []
for i in range(len(ground_truth10[0])):
    for j in range(200):
        committee = Find_Cols_With_n_Conflicts(j+1, j, ground_truth10, i)
        if len(committee) > 0:
            col_dict[i] = j
            if j > 15:
                col_delete.append(i)
            break
ground_truth10 = np.delete(ground_truth10, col_delete, axis=1)
percentages = []
for k in range(1, 5):
    percentages.append(np.sum(np.where(ground_truth10==k, 1, 0)))
plt.bar(range(1, 5), percentages)
plt.show()
np.save("experimental-space-v10-modified", ground_truth10)

print(0)
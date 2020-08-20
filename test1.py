from sklearn.linear_model import LogisticRegression
import math
import numpy as np
import cvxpy as cp
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering


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
data = pd.read_csv(r"compound-target-datasets\target-compound dataset v3.csv", "r", delimiter=",",  engine='python').values
# data = pd.read_csv(r"C:\Users\57539\Downloads\DtcDrugTargetInteractions.csv", "rb", delimiter=",").values
# keys = ['KD-inverse_agonist', 'KI-inhibition-competitive_inhibitor', 
#     'ACTIVITY-inhibition', 'ACTIVITY-inhibition-competitive_inhibitor', 'KD-inhibition-competitive_inhibitor',
#     'IC50-inhibition-competitive_inhibitor', 'IC50-inhibition', 'INHIBITION-inhibition-competitive_inhibitor', 
#     'EC50-activation-competitive_inhibitor', 'EC50-inhibition-competitive_inhibitor', 'INHIBITION-inhibition', 
#     'IC50-inhibition-non_competitive_inhibitor', 'ACTIVITY-inhibition-allosteric_inhibitor', 
#     'KD-inhibition-allosteric_inhibitor', 'IC50-inhibition-allosteric_inhibitor']
# keys = ['inverse_agonist - biochemical - binding - binding_reversible -  - ', 'inhibition - biochemical - functional - enzyme_activity - competitive_inhibitor', 
# 'inhibition - biochemical - functional - enzyme_activity -  - ', 'inhibition - physiochemical - binding - binding_reversible - competitive_inhibitor', 
# 'inhibition - cell_free - functional - enzyme_activity - competitive_inhibitor', 'inhibition - cell_based - functional - enzyme_activity - competitive_inhibitor', 
# 'activation - cell_based - phenotypic - process - competitive_inhibitor', 'inhibition - biochemical - functional - binding_reversible - competitive_inhibitor', 
# 'inhibition - biochemical - functional - binding_saturation - competitive_inhibitor', 'inhibition - physiochemical - binding - binding_reversible - allosteric_inhibitor']
keys = ['inverse_agonist - biochemical - binding - binding_reversible -  - ', 'inhibition - biochemical - functional - enzyme_activity - competitive_inhibitor', 'inhibition - biochemical - functional - enzyme_activity -  - ', 
'inhibition - biochemical - functional - enzyme_activity - non_competitive_inhibitor', 'inhibition - biochemical - functional - binding_saturation - competitive_inhibitor']
phenotypes = {}
assays = {}
print(len(data))
for row in data:
    compound = row[0]
    target = row[4]
    key = compound + ", " + target
    phenotypeA = row[15] + " - "
    phenotypeB = row[16] + " - "
    phenotypeC = row[17]
    phenotypeD = row[18]
    phenotypeE = row[19]
    if not isinstance(phenotypeC, str):
        phenotypeC = " - "
    else: 
        phenotypeC += " - "
    if not isinstance(phenotypeD, str):
        phenotypeD = " - "
    else: 
        phenotypeD += " - "
    if not isinstance(phenotypeE, str):
        phenotypeE = " - "


    phenotype = phenotypeA + phenotypeB + phenotypeC + phenotypeD + phenotypeE
    value = row[12]
    if phenotype not in phenotypes:
        phenotypes[phenotype] = 1
    else:
        phenotypes[phenotype] += 1

    if key in assays:
        if phenotype in assays[key]:
            assays[key][phenotype].append(value)
        else:
            assays[key][phenotype] = []
        assays[key][phenotype].append(value)
    else:
        assays[key] = {}
        assays[key][phenotype] = []
        assays[key][phenotype].append(value)
print(phenotypes.keys())
X = []
targets = []
compounds = []
keys2 = []
for key in assays:
    keys2.append(key)
    temp = key.split(", ")
    if temp[0] not in compounds:
        compounds.append(temp[0])
    if temp[1] not in targets:
        targets.append(temp[1])
    dict_ = assays[key]
    vec = np.empty((5, ))
    vec[:] = np.NaN
    for i in range(5):
        key_ = keys[i]
        if key_ in dict_:
            vec[i] = np.mean(dict_[key_])
    assays[key] = vec
    X.append(vec)

# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
all_ = {}
delete = []
for i in range(len(X)):
    temp = tuple(np.where(np.isnan(X[i]))[0])
    if 2 in temp:
        delete.append(i)
X = np.delete(X, delete, axis = 0)
keys2 = np.delete(keys2, delete, axis = 0)
X = X[:, 2].reshape(-1, 1)

loss = []
# for k in np.arange(1, 31):
#     km = KMeans(n_clusters=k).fit(X)
#     loss.append(km.inertia_)
# plt.plot(np.arange(1, 31), loss, 'o-')
# plt.ylim(0, 10000000)
# plt.show()
# km = KMeans(n_clusters=4, random_state=42).fit(X)
medium = np.median(X)
for i in range(len(X)):
    if  X[i] <= 25:
        X[i] = 1
    elif 25 < X[i] <= 75:
        X[i] = 2
    elif 75 < X[i] <= 125:
        X[i] = 3
    elif 125 < X[i]:
        X[i] = 4




# X += 1
# X = AgglomerativeClustering(distance_threshold=5, n_clusters=None).fit_predict(X)
percentages = []

matrix = -np.ones((len(targets), len(compounds)))
for i in range(len(keys2)):
    temp = keys2[i].split(", ")
    compound = temp[0]
    target = temp[1]
    idx2 = compounds.index(compound)
    idx1 = targets.index(target)
    matrix[idx1, idx2] = X[i]
delete_rows = []
delete_cols = []
# for i in range(matrix.shape[0]):
#     if np.sum(matrix[i]) <= 0 :
#         delete_rows.append(i)
# matrix = np.delete(matrix, delete_rows, axis=0)
# targets = [i for j, i in enumerate(targets) if j not in delete_rows]
# for i in range(matrix.shape[1]):
#     if np.sum(matrix[:, i]) <= 0:
#         delete_cols.append(i)
for i in range(matrix.shape[0]):
    if np.sum(np.where(matrix[i] != -1, 1, 0)) <= 0.4 * len(matrix[0]) :
        delete_rows.append(i)
matrix = np.delete(matrix, delete_rows, axis=0)
targets = [i for j, i in enumerate(targets) if j not in delete_rows]
for i in range(matrix.shape[1]):
    if np.sum(np.where(matrix[:, i] != -1, 1, 0)) <= 0.6 * len(matrix):
        delete_cols.append(i)
matrix = np.delete(matrix, delete_cols, axis=1)
compounds = [i for j, i in enumerate(compounds) if j not in delete_cols]
all_num = np.sum(np.sum(np.where(matrix !=-1, 1, 0)))/(matrix.shape[0]*matrix.shape[1])
u, s, v = np.linalg.svd(matrix)
rank = np.sum(s > 0.01*max(s))
i = 1
for k in range(1, 5):
    percentages.append(np.sum(np.where(matrix==k, 1, 0)))
# print(percentages)
plt.bar(range(1, 5), percentages)
plt.ylabel("percentage")
plt.xlabel("phenotypes")
plt.show()
total = np.sum(np.where(matrix != -1, 1, 0))
np.save("compound-target-datasets\experimental-space-v12", matrix)
row_dict = {}
row_delete = []
for i in range(len(matrix)):
    for j in range(200):
        committee = Find_Cols_With_n_Conflicts(j+1, j, matrix.T, i)
        if len(committee) > 0:
            row_dict[i] = j
            if j > 15:
                row_delete.append(i)
            break
matrix = np.delete(matrix, row_delete, axis=0)
targets = [i for j, i in enumerate(targets) if j not in row_delete]
col_dict = {}
col_delete = []
for i in range(len(matrix[0])):
    for j in range(200):
        committee = Find_Cols_With_n_Conflicts(j+1, j, matrix, i)
        if len(committee) > 0:
            col_dict[i] = j
            if j > 5:
                col_delete.append(i)
            break
matrix = np.delete(matrix, col_delete, axis=1)
percentages = []
for k in range(1, 5):
    percentages.append(np.sum(np.where(matrix==k, 1, 0)))
plt.bar(range(1, 5), percentages)
plt.show()
np.save("compound-target-datasets\experimental-space-v11-modified", matrix)

compounds = [i for j, i in enumerate(compounds) if j not in col_delete]
i = 1
with open(r'compound-target-datasets\compounds_list_v11.txt', 'w') as f: 
    for line in compounds:
        f.write(str(i)+". "+line+"\n") 
        i += 1
i = 1
with open(r'compound-target-datasets\targets_list_v11.txt', 'w') as f: 
    for line in targets:
        f.write(str(i)+". "+line+"\n") 
        i += 1
print(phenotypes.keys())


print(1)
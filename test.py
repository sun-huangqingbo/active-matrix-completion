from sklearn.linear_model import LogisticRegression
import math
import numpy as np
import cvxpy as cp
import pandas as pd
import math
import csv

np.random.seed(10)
data = pd.read_csv(r"C:\Users\57539\Downloads\DtcDrugTargetInteractions.csv", "rb", delimiter=",").values
compound_names = {}
target_names = {}
phenotypes = {}
for row in data:
    compound = row[0]
    target = row[4]
    if row[7] != "wild_type":
        continue
    phenotypeA = str(row[15]) + " - "
    phenotypeB = str(row[16]) + " - "
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
    if phenotype in phenotypes:
        phenotypes[phenotype] += 1
    else: phenotypes[phenotype] = 1
    if phenotype != "inhibition - biochemical - functional - enzyme_activity - ":
        continue
    unit = row[13]
    if unit != "NM":
        continue
    if compound not in compound_names:
        compound_names[compound] = 1
    else: compound_names[compound] += 1
    if target not in target_names:
        target_names[target] = 1
    else: target_names[target] += 1
        

sorted_target = sorted(target_names.items(), key=lambda x: x[1], reverse=True)
sorted_compound = sorted(compound_names.items(), key=lambda x: x[1], reverse=True)
candidate_compounds = []
i = 0
ii = 0
while ii < min(2000, len(sorted_compound)):
    if isinstance(sorted_compound[ii][0], str):
        temp = sorted_compound[ii][0].split(", ")
        if len(temp) > 1:
            for item in temp:
                candidate_compounds.append(item)
                i += 1
        else:
            candidate_compounds.append(temp[0])
            i += 1
        ii+=1
    else: ii +=1
        
candidate_targets = []
i = 0
ii = 0
while ii < min(2000, len(sorted_target)):
    if isinstance(sorted_target[ii][0], str):
        temp = sorted_target[ii][0].split(", ")
        if len(temp) > 1:
            for item in temp:
                if item in candidate_targets:
                    pass
                else:
                    candidate_targets.append(item)
                    i += 1
        else:
            if temp[0] in candidate_targets:
                pass
            else:
                candidate_targets.append(temp[0])
                i += 1
        ii+=1
    else: ii +=1

matrix = np.zeros((len(candidate_targets), len(candidate_compounds)))
for row in data:
    compound = row[0]
    target = row[4]
    # if row[7] != "wild_type":
    #     continue
    if compound in candidate_compounds and target in candidate_targets:
        idx2 = candidate_compounds.index(compound)
        idx1 = candidate_targets.index(target)
        matrix[idx1][idx2] = 1
all_num = np.sum(matrix)
delete_rows = []
delete_cols = []
for i in range(matrix.shape[0]):
    if np.sum(matrix[i]) <= 0.1*matrix.shape[1] :
        delete_rows.append(i)
matrix = np.delete(matrix, delete_rows, axis=0)
candidate_targets = [i for j, i in enumerate(candidate_targets) if j not in delete_rows]
for i in range(matrix.shape[1]):
    if np.sum(matrix[:, i]) <= 0.1*matrix.shape[0]:
        delete_cols.append(i)
matrix = np.delete(matrix, delete_cols, axis=1)
candidate_compounds = [i for j, i in enumerate(candidate_compounds) if j not in delete_cols]
all_num = np.sum(matrix)/(matrix.shape[0]*matrix.shape[1])
new = []
for row in data:
    compound = row[0]
    target = row[4]
    # if row[7] != "wild_type":
    #     continue
    phenotypeA = str(row[15]) + " - "
    phenotypeB = str(row[16]) + " - "
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
    unit = row[13]
    if unit != "%":
        continue
    # if phenotype != 'inhibition - biochemical - functional - enzyme_activity -  - ':
    #     continue
    if phenotype != "inhibition - biochemical - functional - enzyme_activity - ":
        continue
    if compound in candidate_compounds and target in candidate_targets:
        new.append(row)
w = csv.writer(open("target-compound dataset new.csv", "w", encoding='utf-8', newline=''))
w.writerow("")
for row in new:
    w.writerow(row)
print(0)
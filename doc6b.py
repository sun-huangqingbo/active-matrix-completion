import pandas as pd
import numpy as np
import csv
from sklearn.cluster import AgglomerativeClustering   #maybe changed
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def Readin_xls(filename):
    return pd.read_excel(filename, header = 0, index=False)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def Construct_Currdata(measured_samples, targets_num, conditions_num):
    currdata = []
    for i in range(4*targets_num*conditions_num):
        currdata.append('uuu')

    currdata = np.array(currdata).reshape(2*targets_num, 2*conditions_num) 
    '''
    for sample, y in zip(measured_samples, measured_y):
        row = sample[1]-1
        col = sample[0]-1
        currdata[row][col] = groundtruth[row][col]
    '''

    return currdata

def Construct_Groundtruth(all_samples, targets_num, conditions_num, classifier):
    groundtruth = []
    for i in range(4*targets_num*conditions_num):
        groundtruth.append('uuu')

    groundtruth = np.array(groundtruth).reshape(2*targets_num, 2*conditions_num) 
    labels = classifier.predict(all_samples[:,4:])
    for sample, label in zip(all_samples, labels):
        row = sample[1]-1
        col = sample[0]-1
        groundtruth[row][col] = str(label)
        groundtruth[row + targets_num][col] = str(label)
        groundtruth[row][col + conditions_num] = str(label)
        groundtruth[row + targets_num][col + conditions_num] = str(label)
    return groundtruth

def Update_currdata(currdata, groundtruth):
    for i in range(len(currdata)):
        for j in range(len(currdata[0])):
            if currdata[i][j]!= 'uuu':
                currdata[i][j] = groundtruth[i][j]
    return currdata



data = np.array(Readin_xls('E:\Active_Learning\elife\elife-10047-supp4-v2.xls'))


measured_samples = np.array(data[:1000])
cluster_num = []

clustering = AgglomerativeClustering(distance_threshold=9, n_clusters=None).fit(measured_samples[:, 4:])
measured_y = clustering.labels_


#forest = RandomForestClassifier(n_estimators=int(0.1 * len(measured_samples)), max_depth = 5).fit(measured_samples[:, 4:], measured_y)

#currdata = Construct_Currdata(measured_samples, measured_y, 48, 46)
#groundtruth = Construct_Groundtruth(data, 48, 46, forest)

print(0)


plot_dendrogram(clustering, p = 115)
plt.show()